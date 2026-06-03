"""
demo_wing_analytics -- turn 3D flow into the numbers: a finite-wing polar.

Sweeps a 3D wing (imported STL) through angle of attack on the fast CUDA solver
and reports the engineering coefficients RC builders actually decide on:

    Cl, Cd, L/D   vs angle of attack

Unlike the 2D airfoil sweep, this is a *finite* wing: the tip vortices bleed off
lift and add induced drag, so the lift slope is lower and drag higher than 2D --
the real behaviour of a stubby low-aspect-ratio wing. Coefficients use the wing
planform area.

This is the bridge from "see the flow" to "use the numbers", and the machinery
that will later be pointed at real UIUC wind-tunnel data.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp

from lbm3d_cuda import LBM3D_CUDA
from stl_import import (naca_wing, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize)
from demo_stl_flow import rotate_aoa

OUT = os.path.join(os.path.dirname(__file__), "out")


def run_aoa(stl, aoa, nx, ny, nz, re, u_lb, steps):
    tris = rotate_aoa(load_binary_stl(stl), aoa)
    mask = voxelize(fit_to_grid(tris, nx, ny, nz, 0.22), nx, ny, nz)
    sim = LBM3D_CUDA(nx, ny, nz, u_lb=u_lb, re=re, char_length=nx * 0.3)
    sim.set_solid(mask)
    for _ in range(steps):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    area = float(mask.any(axis=1).sum())          # planform area (project on x-z)
    F = sim.compute_force()
    denom = 0.5 * u_lb ** 2 * area
    cd, cl = F[0] / denom, F[1] / denom           # flow +x: Fx drag, Fy lift
    return cl, cd, area


def main():
    os.makedirs(OUT, exist_ok=True)
    stl = os.path.join(OUT, "test_wing.stl")
    if not os.path.exists(stl):
        write_binary_stl(stl, naca_wing(1.0, 1.6, 0.12))

    nx, ny, nz = 170, 100, 130
    # Re=600 keeps plain BGK stable (tau~0.513) at this resolution; pushing to
    # RC Reynolds (1e4+) needs a cumulant/regularised collision operator -- the
    # known next step (see docs/VALIDATION.md).
    re, u_lb, steps = 600.0, 0.05, 16000
    alphas = np.array([0, 3, 6, 9, 12], dtype=float)
    ar = 1.6                                       # span/chord of the test wing

    print(f"3D finite-wing polar  {nx}x{ny}x{nz}  Re={re:.0f}  AR~{ar}  "
          f"(CUDA solver)", flush=True)
    cls, cds = [], []
    for a in alphas:
        cl, cd, area = run_aoa(stl, a, nx, ny, nz, re, u_lb, steps)
        cls.append(cl); cds.append(cd)
        print(f"  AoA={a:4.1f}  Cl={cl:+.3f}  Cd={cd:.3f}  "
              f"L/D={cl/cd if cd else 0:+5.1f}", flush=True)

    cls, cds = np.array(cls), np.array(cds)
    np.savez(os.path.join(OUT, "wing_polar3d.npz"), alpha=alphas, cl=cls, cd=cds)
    _report(alphas, cls, cds, ar)
    _plot(alphas, cls, cds)


def _report(alphas, cls, cds, ar):
    lin = alphas <= 9.0
    slope = np.polyfit(alphas[lin], cls[lin], 1)[0] * 180.0 / np.pi
    i_best = int(np.argmax(cls / np.where(cds > 0, cds, 1e9)))
    print("\n" + "=" * 58)
    print("  3D FINITE-WING POLAR")
    print("=" * 58)
    print(f"  lift-curve slope     = {slope:.2f} /rad")
    print(f"  (2D thin-airfoil 2pi=6.28; finite low-AR={ar} sits well below)")
    print(f"  best L/D             = {cls[i_best]/cds[i_best]:.1f} "
          f"at AoA={alphas[i_best]:.0f}")
    print(f"  max Cl / stall       = {cls.max():.3f} at "
          f"{alphas[int(np.argmax(cls))]:.0f} deg")
    print("=" * 58)
    print("  Finite-wing physics: tip vortices reduce lift slope and add")
    print("  induced drag vs a 2D section -- as expected for a real wing.")
    print("=" * 58)


def _plot(alphas, cls, cds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(alphas, cls, "o-"); ax[0].set_xlabel("AoA (deg)")
    ax[0].set_ylabel("Cl"); ax[0].set_title("lift curve"); ax[0].grid(alpha=.3)
    ax[1].plot(alphas, cds, "o-", color="C3"); ax[1].set_xlabel("AoA (deg)")
    ax[1].set_ylabel("Cd"); ax[1].set_title("drag"); ax[1].grid(alpha=.3)
    ax[2].plot(cds, cls, "o-", color="C2"); ax[2].set_xlabel("Cd")
    ax[2].set_ylabel("Cl"); ax[2].set_title("drag polar"); ax[2].grid(alpha=.3)
    fig.suptitle("3D finite-wing polar (CUDA solver, with tip effects)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wing_polar3d.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
