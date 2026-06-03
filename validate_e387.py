"""
validate_e387 -- the trust milestone: compare against REAL wind-tunnel data.

The Eppler E387 is the canonical low-Reynolds validation airfoil, with UIUC
wind-tunnel measurements. We run the actual E387 geometry (real coordinates) at
a real RC Reynolds number with the LES operator and compare lift and drag to the
measured reference.

Reference (UIUC E387, well-established; measured anchor Cl=0.47 at alpha=1 deg):
    zero-lift angle ~ -3.5 deg,  Cl at 0 deg ~ 0.40,  lift slope ~ 0.10 /deg,
    Cd_min ~ 0.01 (Re=2e5).  (The E387 is CAMBERED -> positive lift at 0 deg.)

HONEST EXPECTATION: lift is circulation-dominated and should track the data;
drag at this Reynolds number needs boundary-layer resolution we cannot afford in
2D, so Cd is expected to be over-predicted. Reported either way -- the point is
to find out, truthfully, how close we get.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp
from matplotlib.path import Path

from lbm2d_cuda import LBM2D_CUDA

OUT = os.path.join(os.path.dirname(__file__), "out")
DATA = os.path.join(os.path.dirname(__file__), "data", "e387.dat")

# Established E387 reference (UIUC, Re~1e5-2e5).
REF = dict(cl0=0.40, slope_deg=0.10, cl_at_1=0.47, alpha0=-3.5, cd_min=0.012)


def load_airfoil():
    pts = []
    with open(DATA) as fh:
        for line in fh:
            try:
                x, y = line.split()
                pts.append((float(x), float(y)))
            except ValueError:
                continue
    return np.array(pts)


def airfoil_mask(coords, nx, ny, chord, cx, cy, aoa_deg):
    a = np.radians(-aoa_deg)
    p = coords.copy()
    p[:, 0] -= 0.25                        # pivot about the quarter chord
    p = p * chord
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = p @ rot.T + np.array([cx, cy])
    path = Path(p)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = path.contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
    return inside.reshape(nx, ny)


def run_aoa(coords, aoa, nx, ny, chord, re, u_lb, steps, sample=300):
    sim = LBM2D_CUDA(nx, ny, u_lb=u_lb, re=re, char_length=chord,
                     collision="les")
    mask = airfoil_mask(coords, nx, ny, chord, 1.5 * chord, ny / 2.0, aoa)
    sim.set_solid(mask)
    cl, cd = [], []
    start = int(0.55 * steps)
    for t in range(steps):
        sim.step()
        if t >= start and t % sample == 0:        # force is off the hot loop
            d, l = sim.coefficients()
            cd.append(d); cl.append(l)
    cp.cuda.Stream.null.synchronize()
    if not bool(cp.all(cp.isfinite(sim.f_a))):
        return float("nan"), float("nan")
    return float(np.mean(cl)), float(np.mean(cd))


def main():
    coords = load_airfoil()
    # High resolution (native CUDA kernel makes this affordable): chord=600 ->
    # the ~3% camber is ~18 cells, resolvable, where chord=130 had only ~4.
    nx, ny = 2400, 1400
    chord = 600.0
    re, u_lb, steps = 100000.0, 0.05, 110000
    alphas = np.array([-2, 0, 2, 4, 6, 8], dtype=float)

    print(f"E387 vs UIUC data  {nx}x{ny}  chord={chord:.0f}  Re={re:.0f}  "
          f"LES (CUDA kernel)", flush=True)
    cls, cds = [], []
    for a in alphas:
        cl, cd = run_aoa(coords, a, nx, ny, chord, re, u_lb, steps)
        cls.append(cl); cds.append(cd)
        print(f"  alpha={a:+4.0f}  Cl={cl:+.3f}  Cd={cd:.4f}", flush=True)

    cls, cds = np.array(cls), np.array(cds)
    np.savez(os.path.join(OUT, "e387_polar.npz"), alpha=alphas, cl=cls, cd=cds)
    _report(alphas, cls, cds)
    _plot(alphas, cls, cds)


def _report(alphas, cls, cds):
    ok = np.isfinite(cls)
    slope = np.polyfit(alphas[ok], cls[ok], 1)[0]
    cl0 = float(np.interp(0.0, alphas, cls))
    alpha0 = float(np.interp(0.0, cls, alphas)) if cls.min() < 0 < cls.max() \
        else float("nan")
    cl1 = float(np.interp(1.0, alphas, cls))
    print("\n" + "=" * 62)
    print("  E387 vs UIUC WIND-TUNNEL DATA")
    print("=" * 62)
    print(f"  {'quantity':22} {'FluidSim':>10} {'UIUC ref':>10}  {'err':>8}")
    rows = [("lift slope (/deg)", slope, REF["slope_deg"]),
            ("Cl at 0 deg", cl0, REF["cl0"]),
            ("Cl at 1 deg", cl1, REF["cl_at_1"]),
            ("zero-lift angle", alpha0, REF["alpha0"]),
            ("Cd min", float(np.nanmin(cds)), REF["cd_min"])]
    for name, got, ref in rows:
        err = (got - ref) / ref * 100 if ref else float("nan")
        print(f"  {name:22} {got:>10.3f} {ref:>10.3f}  {err:>+7.0f}%")
    print("=" * 62)
    print("  Lift: circulation-dominated -- the meaningful comparison.")
    print("  Cd: expect over-prediction (boundary layer under-resolved at")
    print("  this Re in 2D). Honest readout, not a tuned result.")
    print("=" * 62)


def _plot(alphas, cls, cds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aref = np.linspace(-3.5, 8, 50)
    clref = REF["cl0"] + REF["slope_deg"] * aref
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(aref, clref, "k--", label="UIUC ref (linear)")
    ax[0].plot(alphas, cls, "o-", color="C0", label="FluidSim (LES)")
    ax[0].set_xlabel("alpha (deg)"); ax[0].set_ylabel("Cl")
    ax[0].set_title("lift curve vs E387 data"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(cds, cls, "o-", color="C2")
    ax[1].axvline(REF["cd_min"], color="k", ls="--", label="ref Cd_min")
    ax[1].set_xlabel("Cd"); ax[1].set_ylabel("Cl"); ax[1].legend()
    ax[1].set_title("drag polar"); ax[1].grid(alpha=.3)
    fig.suptitle("E387 at Re=1e5: FluidSim (LES) vs UIUC wind-tunnel reference")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e387_validation.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
