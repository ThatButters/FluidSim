"""
demo_stl_flow -- the end-to-end 3D pipeline: STL file -> 3D flow -> picture.

Loads a wing mesh from a binary .stl, voxelises it onto the GPU lattice, runs the
3D D3Q19 solver, and saves slice views of the flow around the imported shape.
This is the proof of the core 3D vision in miniature: an arbitrary mesh becomes a
solid the simulated air flows around -- no analytic geometry, a real file in.

(Generates its own wing STL if one isn't supplied, so it runs standalone.)
"""

from __future__ import annotations

import os
import numpy as np

try:
    import cupy as cp
    XP = cp if cp.cuda.runtime.getDeviceCount() > 0 else np
except Exception:
    cp, XP = None, np

from lbm3d import LBM3D
from stl_import import (naca_wing, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize)

OUT = os.path.join(os.path.dirname(__file__), "out")


def rotate_aoa(tris, deg):
    """Rotate a mesh about its xy-centre to set an angle of attack (+x flow)."""
    a = np.radians(-deg)
    ca, sa = np.cos(a), np.sin(a)
    out = tris.copy()
    c = tris[..., :2].reshape(-1, 2).mean(0)
    rx = tris[..., 0] - c[0]
    ry = tris[..., 1] - c[1]
    out[..., 0] = c[0] + rx * ca - ry * sa
    out[..., 1] = c[1] + rx * sa + ry * ca
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    stl = os.path.join(OUT, "test_wing.stl")
    if not os.path.exists(stl):
        write_binary_stl(stl, naca_wing(chord=1.0, span=1.6, thickness=0.12))

    nx, ny, nz = 180, 110, 130
    aoa = 10.0
    tris = rotate_aoa(load_binary_stl(stl), aoa)
    mask_np = voxelize(fit_to_grid(tris, nx, ny, nz, margin=0.2), nx, ny, nz)
    print(f"Loaded STL -> voxelised: {int(mask_np.sum())} solid cells "
          f"in {nx}x{ny}x{nz}, AoA={aoa:.0f} deg")

    sim = LBM3D(nx, ny, nz, u_lb=0.05, re=800.0, char_length=nx * 0.3,
                array_module=XP, dtype=np.float32 if XP is np else cp.float32)
    sim.solid = XP.asarray(mask_np)
    sim.body = sim.solid
    backend = "GPU" if XP is not np else "CPU"
    print(f"Running 3D flow past the imported wing on {backend} ...", flush=True)

    steps = 4000
    for t in range(steps):
        sim.step()
        if t % 1000 == 0:
            if XP is not np:
                cp.cuda.Stream.null.synchronize()
            fx = float(sim.force[0]); fy = float(sim.force[1])
            print(f"  step {t:5d}/{steps}  Fx(drag)={fx:+.3f}  "
                  f"Fy(lift)={fy:+.3f}", flush=True)

    _slices(sim, mask_np, aoa)
    print(f"\nSaved flow slices to {OUT}/stl_flow_*.png  "
          f"(imported mesh -> 3D flow, end to end).")


def _slices(sim, mask_np, aoa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, u = sim.macroscopic()
    spd = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5
    spd = spd if XP is np else cp.asnumpy(spd)

    # Mid-span slice (x-y): the airfoil section and its wake.
    k = sim.nz // 2
    a = spd[:, :, k].copy(); a[mask_np[:, :, k]] = np.nan
    # Mid-thickness slice (x-z): the planform, wing tips and tip wake.
    j = sim.ny // 2
    b = spd[:, j, :].copy(); b[mask_np[:, j, :]] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(a.T, origin="lower", cmap="viridis")
    ax[0].set_title(f"mid-span section (x-y), AoA={aoa:.0f}"); ax[0].axis("off")
    ax[1].imshow(b.T, origin="lower", cmap="viridis")
    ax[1].set_title("planform / tips (x-z)"); ax[1].axis("off")
    fig.suptitle("3D flow past an imported STL wing (speed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stl_flow_slices.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
