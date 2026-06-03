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
from demo_stl_flow import rotate_aoa


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
        self.aoa = 6.0
        self.steps = 0
        self._tris = None
        self.sim = None
        self.body = None
        self.level = 1e-6

    # -- model loading ------------------------------------------------------
    def load(self, stl_path=None, aoa=6.0):
        if stl_path is None:
            out = os.path.join(os.path.dirname(__file__), "out")
            os.makedirs(out, exist_ok=True)
            stl_path = os.path.join(out, "test_wing.stl")
            if not os.path.exists(stl_path):
                write_binary_stl(stl_path, naca_wing(1.0, 1.6, 0.12))
        self._tris = load_binary_stl(stl_path)
        self.aoa = aoa
        self._rebuild(warmup=1400)
        return int(self.mask.sum())

    def _voxelise(self):
        tris = rotate_aoa(self._tris, self.aoa)
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
    def set_aoa(self, aoa):
        self.aoa = float(aoa)
        self._rebuild(warmup=600)

    def set_reynolds(self, re):
        self.re = float(re)
        nu = self.u_lb * self.char_length / self.re
        self.sim.omega = np.float32(1.0 / (3.0 * nu + 0.5))

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
