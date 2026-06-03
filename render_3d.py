"""
render_3d -- 3D visualisation of the flow: vortex isosurfaces around the body.

Runs the fast CUDA solver around an imported STL wing, then extracts and renders
3D surfaces with marching cubes:

  - the body itself (the voxelised wing), and
  - a vorticity-magnitude isosurface -- the wake sheet and tip vortices, the
    "glowing tubes" of the 3D vision.

This is a still render (matplotlib 3D), not the real-time GPU renderer yet, but
it is the first time the 3D flow *structure* is shown as 3D shapes rather than
slices. The solver speed (fused CUDA kernel) is what makes generating the field
quick.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp
from skimage import measure

from lbm3d_cuda import LBM3D_CUDA
from stl_import import (naca_wing, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize)
from demo_stl_flow import rotate_aoa

OUT = os.path.join(os.path.dirname(__file__), "out")


def q_criterion(u):
    """Q-criterion: 0.5(|Omega|^2 - |S|^2). Positive Q isolates vortex cores
    (rotation-dominated) from shear layers -- gives clean tip-vortex tubes."""
    g = [[np.gradient(u[i], axis=j) for j in range(3)] for i in range(3)]
    q = np.zeros_like(u[0])
    for i in range(3):
        for j in range(3):
            s = 0.5 * (g[i][j] + g[j][i])         # strain rate
            w = 0.5 * (g[i][j] - g[j][i])         # rotation
            q += 0.5 * (w * w - s * s)
    return q


def main():
    os.makedirs(OUT, exist_ok=True)
    stl = os.path.join(OUT, "test_wing.stl")
    if not os.path.exists(stl):
        write_binary_stl(stl, naca_wing(chord=1.0, span=1.6, thickness=0.12))

    nx, ny, nz = 200, 120, 150
    aoa = 8.0
    tris = rotate_aoa(load_binary_stl(stl), aoa)
    mask = voxelize(fit_to_grid(tris, nx, ny, nz, margin=0.22), nx, ny, nz)

    sim = LBM3D_CUDA(nx, ny, nz, u_lb=0.05, re=900.0, char_length=nx * 0.3)
    sim.set_solid(mask)
    steps = 4500
    print(f"Running 3D flow ({nx}x{ny}x{nz}, {int(mask.sum())} solid cells, "
          f"AoA={aoa:.0f}) on the fused CUDA kernel ...", flush=True)
    import time
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    print(f"  {steps} steps in {time.perf_counter()-t0:.1f}s "
          f"({steps/(time.perf_counter()-t0):.0f} steps/s)", flush=True)

    _, u = sim.macroscopic()
    u = cp.asnumpy(u)
    q = q_criterion(u)
    from skimage.filters import gaussian
    q = gaussian(q, sigma=1.2)                    # merge fragments into tubes
    q[mask] = 0.0
    _render(mask, q, aoa)


def _render(mask, q, aoa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Body surface, and a Q-criterion isosurface (coherent vortex cores only).
    bv, bf, _, _ = measure.marching_cubes(mask.astype(np.float32), 0.5)
    level = np.percentile(q[q > 0], 96)           # strong rotation only
    vv, vf, _, _ = measure.marching_cubes(q, level, step_size=2)

    fig = plt.figure(figsize=(16, 6.5))
    for n, (elev, azim) in enumerate([(20, -60), (80, -90)]):
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        ax.plot_trisurf(bv[:, 0], bv[:, 1], bv[:, 2], triangles=bf,
                        color="#222222", alpha=1.0, linewidth=0, shade=True)
        ax.plot_trisurf(vv[:, 0], vv[:, 1], vv[:, 2], triangles=vf,
                        color="#23c0ff", alpha=0.55, linewidth=0,
                        antialiased=False, shade=True)
        ax.set_box_aspect(mask.shape)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("3/4 view" if n == 0 else "top (planform): tip vortices")
    fig.suptitle(f"3D vortex cores around an imported wing  "
                 f"(AoA={aoa:.0f}, Q-criterion isosurface)")
    fig.tight_layout()
    out = os.path.join(OUT, "render_3d.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved 3D render -> {out}")


if __name__ == "__main__":
    main()
