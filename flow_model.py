"""
flow_model -- the simulation + visualisation engine behind the viewers/GUI.

Wraps the fused 3D CUDA solver with STL voxelisation, live vortex-core extraction
(Q-criterion, on the GPU) and force readout, exposing a clean renderer-agnostic
interface (returns PyVista meshes + coefficients) that the desktop GUI drives.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp
import pyvista as pv

from lbm3d_cuda import LBM3D_CUDA
from stl_import import (naca_wing, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize)


def rotate_model(tris, pitch_deg, yaw_deg):
    """Orient a mesh in the (+x) wind: pitch about the span (z) axis -- angle of
    attack -- and yaw about the vertical (y) axis -- turn left/right."""
    p, y = np.radians(-pitch_deg), np.radians(yaw_deg)
    out = tris.astype(np.float32).copy()
    c = out.reshape(-1, 3).mean(0)
    rel = out - c
    x, yy, z = rel[..., 0].copy(), rel[..., 1].copy(), rel[..., 2].copy()
    # pitch (x-y plane)
    xp = x * np.cos(p) - yy * np.sin(p)
    yp = x * np.sin(p) + yy * np.cos(p)
    x, yy = xp, yp
    # yaw (x-z plane)
    xq = x * np.cos(y) + z * np.sin(y)
    zq = -x * np.sin(y) + z * np.cos(y)
    out[..., 0] = xq + c[0]
    out[..., 1] = yy + c[1]
    out[..., 2] = zq + c[2]
    return out


def q_criterion_gpu(u):
    """Q-criterion (vortex cores) on the GPU; returns a host float32 array."""
    g = [[cp.gradient(u[i], axis=j) for j in range(3)] for i in range(3)]
    q = cp.zeros_like(u[0])
    for i in range(3):
        for j in range(3):
            s = 0.5 * (g[i][j] + g[j][i])
            w = 0.5 * (g[i][j] - g[j][i])
            q += 0.5 * (w * w - s * s)
    return cp.asnumpy(q)


class FlowModel:
    def __init__(self, nx=150, ny=96, nz=120, re=600.0, u_lb=0.05):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.re, self.u_lb = re, u_lb
        self.char_length = nx * 0.3
        self.pitch = 6.0
        self.yaw = 0.0
        self.steps = 0
        self._tris = None
        self.sim = None
        self.body = None
        self.level = 1e-6

    # -- model loading ------------------------------------------------------
    def load(self, stl_path=None, pitch=6.0, yaw=0.0):
        if stl_path is None:
            out = os.path.join(os.path.dirname(__file__), "out")
            os.makedirs(out, exist_ok=True)
            stl_path = os.path.join(out, "test_wing.stl")
            if not os.path.exists(stl_path):
                write_binary_stl(stl_path, naca_wing(1.0, 1.6, 0.12))
        self._tris = load_binary_stl(stl_path)
        self.pitch, self.yaw = pitch, yaw
        self._rebuild(warmup=1400)
        return int(self.mask.sum())

    def _voxelise(self):
        tris = rotate_model(self._tris, self.pitch, self.yaw)
        self.mask = voxelize(fit_to_grid(tris, self.nx, self.ny, self.nz, 0.22),
                             self.nx, self.ny, self.nz)
        self.area = float(self.mask.any(axis=1).sum()) or 1.0

    def _rebuild(self, warmup=0):
        self._voxelise()
        self.sim = LBM3D_CUDA(self.nx, self.ny, self.nz, u_lb=self.u_lb,
                              re=self.re, char_length=self.char_length,
                              collision="les")
        self.sim.set_solid(self.mask)
        self.body = self._image(self.mask.astype(np.float32)).contour([0.5])
        self.steps = 0
        for _ in range(warmup):
            self.sim.step()
        self.level = self._auto_level()

    # -- live controls ------------------------------------------------------
    def set_pitch(self, pitch):
        self.pitch = float(pitch)
        self._rebuild(warmup=600)

    def set_yaw(self, yaw):
        self.yaw = float(yaw)
        self._rebuild(warmup=600)

    def set_orientation(self, pitch, yaw):
        """Set pitch and yaw together (one rebuild) -- used by mouse drag."""
        self.pitch, self.yaw = float(pitch), float(yaw)
        self._rebuild(warmup=600)

    def set_reynolds(self, re):
        self.re = float(re)
        nu = self.u_lb * self.char_length / self.re
        self.sim.omega = np.float32(1.0 / (3.0 * nu + 0.5))

    def relevel(self):
        """Re-pick the vortex isosurface level (after a wind-speed change)."""
        self.level = self._auto_level()

    def reset(self):
        self._rebuild(warmup=600)

    def step(self, n):
        for _ in range(n):
            self.sim.step()
        self.steps += n

    # -- outputs ------------------------------------------------------------
    def _image(self, field):
        grid = pv.ImageData(dimensions=(self.nx, self.ny, self.nz))
        grid.point_data["v"] = field.ravel(order="F")
        return grid

    def _q_field(self):
        _, u = self.sim.macroscopic()
        q = q_criterion_gpu(u)
        q[self.mask] = 0.0
        return q

    def _auto_level(self):
        q = self._q_field()
        pos = q[q > 0]
        return float(np.percentile(pos, 90)) if pos.size else 1e-6

    def vortex_mesh(self):
        return self._image(self._q_field()).contour([self.level], scalars="v")

    def coefficients(self):
        F = self.sim.compute_force()
        denom = 0.5 * self.u_lb ** 2 * self.area
        cl, cd = F[1] / denom, F[0] / denom
        ld = cl / cd if abs(cd) > 1e-9 else 0.0
        return cl, cd, ld

    def is_finite(self):
        return bool(cp.all(cp.isfinite(self.sim.f_a)))
