"""
live_viewer_3d -- the real-time interactive 3D wind tunnel.

An imported wing sits in a live 3D flow on the GPU. The fused CUDA solver steps
continuously; each update re-extracts the vortex cores (Q-criterion, computed on
the GPU) and the renderer shows them as 3D isosurfaces around the body. You orbit
and zoom with the mouse while the flow evolves -- the "fly around it" experience.

Run it yourself (opens an interactive window):

    python live_viewer_3d.py
    python live_viewer_3d.py --n 120 80 100   # smaller/faster domain
    python live_viewer_3d.py --aoa 12

Controls:
    mouse drag    orbit            mouse wheel   zoom
    space         pause / resume   r             reset the flow
    q  or close   quit

The solver and Q-criterion run on the GPU; only the (single) scalar field is
brought to the host each update for VTK to contour, so the camera stays smooth.
"""

from __future__ import annotations

import argparse
import numpy as np
import cupy as cp
import pyvista as pv

from lbm3d_cuda import LBM3D_CUDA
from stl_import import (naca_wing, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize)
from demo_stl_flow import rotate_aoa
import os


def q_criterion_gpu(u):
    """Q-criterion on the GPU; returns a host float32 array."""
    g = [[cp.gradient(u[i], axis=j) for j in range(3)] for i in range(3)]
    q = cp.zeros_like(u[0])
    for i in range(3):
        for j in range(3):
            s = 0.5 * (g[i][j] + g[j][i])
            w = 0.5 * (g[i][j] - g[j][i])
            q += 0.5 * (w * w - s * s)
    return cp.asnumpy(q)


class Viewer3D:
    def __init__(self, nx, ny, nz, aoa, spf=8, stl_path=None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.aoa, self.spf = aoa, spf
        self.paused = False
        self.steps = 0

        out = os.path.join(os.path.dirname(__file__), "out")
        os.makedirs(out, exist_ok=True)
        if stl_path is None:                          # built-in demo wing
            stl_path = os.path.join(out, "test_wing.stl")
            if not os.path.exists(stl_path):
                write_binary_stl(stl_path, naca_wing(1.0, 1.6, 0.12))
        print(f"Loading {os.path.basename(stl_path)} ...", flush=True)
        tris = rotate_aoa(load_binary_stl(stl_path), aoa)
        self.mask = voxelize(fit_to_grid(tris, nx, ny, nz, 0.22), nx, ny, nz)
        if int(self.mask.sum()) == 0:
            raise SystemExit("Voxelisation produced an empty model -- is the "
                             "STL a valid binary mesh?")
        self.area = float(self.mask.any(axis=1).sum())   # planform (for Cl/Cd)

        self._build()
        # Body surface (static) and a Q level taken from a developed warm-up
        # (the fused kernel is fast, so this is a couple of seconds).
        self.body = self._image(self.mask.astype(np.float32)).contour([0.5])
        print("Warming up the flow ...", flush=True)
        for _ in range(1500):
            self.sim.step()
        self.steps = 0
        self.level = self._auto_level()

    def _build(self):
        # Re=600 keeps plain BGK stable at this resolution (higher Re needs a
        # cumulant collision operator -- the documented next step).
        self.sim = LBM3D_CUDA(self.nx, self.ny, self.nz, u_lb=0.05, re=600.0,
                              char_length=self.nx * 0.3)
        self.sim.set_solid(self.mask)
        self.steps = 0

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
        return float(np.percentile(q[q > 0], 90))

    def _vortex_mesh(self):
        return self._image(self._q_field()).contour([self.level], scalars="v")

    # -- interactive run ----------------------------------------------------
    def _refresh(self):
        """Re-extract the vortex isosurface + live coefficients (HUD)."""
        self.pl.add_mesh(self._vortex_mesh(), name="vortex", color="#23c0ff",
                         opacity=0.45, smooth_shading=True)
        F = self.sim.compute_force()
        denom = 0.5 * self.sim.u_lb ** 2 * self.area
        cd, cl = F[0] / denom, F[1] / denom
        ld = cl / cd if abs(cd) > 1e-9 else 0.0
        self.pl.add_text(
            f"AoA {self.aoa:.0f} deg   steps {self.steps}\n"
            f"Cl {cl:+.3f}   Cd {cd:.3f}   L/D {ld:+.1f}"
            + ("   [PAUSED]" if self.paused else ""),
            name="hud", position="upper_left", font_size=10, color="white")

    def run(self):
        self.pl = pv.Plotter(title="FluidSim -- live 3D wind tunnel")
        self.pl.set_background("#0b0f1a")
        self.pl.add_mesh(self.body, color="#d0d0d0", smooth_shading=True)
        self._refresh()
        self.pl.add_key_event("space", self._toggle)
        self.pl.add_key_event("r", self._reset)
        self.pl.add_callback(self._tick, interval=30)
        self.pl.camera_position = "yz"
        self.pl.camera.azimuth = 35
        self.pl.camera.elevation = 20
        print("Launching live 3D viewer. Orbit with the mouse; q or close to "
              "quit.")
        self.pl.show()

    def _tick(self, *_):
        if not self.paused:
            for _ in range(self.spf):
                self.sim.step()
            self.steps += self.spf
        self._refresh()

    def _toggle(self):
        self.paused = not self.paused

    def _reset(self):
        self._build()


def main():
    p = argparse.ArgumentParser(description="FluidSim live 3D viewer")
    p.add_argument("stl", nargs="?", default=None,
                   help="path to a binary .stl model (omit for the demo wing)")
    p.add_argument("--n", type=int, nargs=3, default=[150, 96, 120],
                   metavar=("NX", "NY", "NZ"))
    p.add_argument("--aoa", type=float, default=9.0)
    args = p.parse_args()
    v = Viewer3D(args.n[0], args.n[1], args.n[2], args.aoa, stl_path=args.stl)
    v.run()


if __name__ == "__main__":
    main()
