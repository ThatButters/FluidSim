"""
demo_airfoil -- first RC-relevant validation: a 2D airfoil lift/drag polar.

A NACA 0012 airfoil is placed in uniform flow and swept through angle of attack.
For each angle we extract the lift and drag coefficients (via the validated
momentum-exchange force) and assemble the polar:

  - zero-lift at alpha = 0 (symmetry check)
  - linear lift region with slope compared to thin-airfoil theory (2*pi/rad)
  - a drag "bucket" (minimum near alpha = 0, rising with |alpha|)
  - stall at high alpha

HONEST SCOPE: plain BGK on a CPU cannot *stably* reach true RC Reynolds numbers
(60k+, where tau -> 1/2). This runs at the highest Reynolds number we can reach
(~1e3) and validates the airfoil aerodynamics against theory. Matching the UIUC
low-Re wind-tunnel *polars* at Re = 60k+ is a GPU-era milestone (it needs both
the resolution and a cumulant/regularised collision operator). This harness is
exactly what we will point at that data once the solver can reach it.
"""

from __future__ import annotations

import os
import numpy as np

from lbm2d import LBM2D

OUT = os.path.join(os.path.dirname(__file__), "out")


def naca0012_mask(nx, ny, x0, y0, chord, alpha_deg):
    """Solid mask for a NACA 0012 at angle of attack alpha (degrees).

    Positive alpha pitches the leading edge up relative to the +x freestream
    (so positive alpha gives positive lift, +y).
    """
    a = -np.radians(alpha_deg)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx, dy = X - x0, Y - y0
    # Rotate lab point into the chord-aligned airfoil frame.
    xc = (dx * np.cos(a) + dy * np.sin(a)) / chord          # 0..1 along chord
    yc = (-dx * np.sin(a) + dy * np.cos(a)) / chord          # across chord
    t = 0.12                                                 # 12% thickness
    with np.errstate(invalid="ignore"):
        yt = 5 * t * (0.2969 * np.sqrt(np.clip(xc, 0, None))
                      - 0.1260 * xc - 0.3516 * xc ** 2
                      + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
    return (xc >= 0.0) & (xc <= 1.0) & (np.abs(yc) <= yt)


def run_one(alpha_deg, nx, ny, chord, re, u_lb, steps):
    sim = LBM2D(nx, ny, u_lb=u_lb, re=re, char_length=chord)
    mask = naca0012_mask(nx, ny, x0=1.4 * chord, y0=ny / 2.0,
                         chord=chord, alpha_deg=alpha_deg)
    sim.solid = mask
    sim.body = mask
    cl_hist = np.empty(steps)
    cd_hist = np.empty(steps)
    for t in range(steps):
        sim.step()
        fx, fy = sim.force
        denom = 0.5 * u_lb ** 2 * chord
        cd_hist[t] = fx / denom                              # drag (along flow)
        cl_hist[t] = fy / denom                              # lift (across flow)
    if not np.all(np.isfinite(cl_hist)):
        return float("nan"), float("nan")
    tail = slice(int(0.6 * steps), None)                     # converged window
    return float(cl_hist[tail].mean()), float(cd_hist[tail].mean())


def main():
    chord = 90.0
    nx, ny = 400, 260
    re = 1000.0
    u_lb = 0.05
    steps = 13000
    alphas = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)

    print(f"NACA0012 polar  {nx}x{ny}  chord={chord:.0f}  Re={re:.0f}  "
          f"{len(alphas)} angles x {steps} steps", flush=True)
    cls, cds = [], []
    for al in alphas:
        cl, cd = run_one(al, nx, ny, chord, re, u_lb, steps)
        cls.append(cl); cds.append(cd)
        print(f"  alpha={al:4.1f}  Cl={cl:+.3f}  Cd={cd:.3f}  "
              f"L/D={cl/cd if cd else 0:+.1f}", flush=True)

    cls, cds = np.array(cls), np.array(cds)
    np.savez(os.path.join(OUT, "airfoil_polar.npz"),
             alpha=alphas, cl=cls, cd=cds)
    _report(alphas, cls, cds)
    _plot(alphas, cls, cds)


def _report(alphas, cls, cds):
    # Lift-curve slope from the linear (attached) region, alpha <= 6 deg.
    lin = alphas <= 6.0
    slope_per_deg = np.polyfit(alphas[lin], cls[lin], 1)[0]
    slope_per_rad = slope_per_deg * 180.0 / np.pi
    cl0 = float(np.interp(0.0, alphas, cls))
    stall = alphas[int(np.argmax(cls))]

    print("\n" + "=" * 60)
    print("  NACA 0012 POLAR -- airfoil aerodynamics validation")
    print("=" * 60)
    print(f"  zero-lift Cl(alpha=0)   = {cl0:+.3f}   (expect ~0, symmetric)")
    print(f"  lift-curve slope        = {slope_per_rad:.2f} /rad "
          f"({slope_per_deg:.3f} /deg)")
    print(f"  thin-airfoil theory     = {2*np.pi:.2f} /rad (inviscid ceiling; "
          f"viscous low-Re sits below)")
    print(f"  max Cl / stall alpha    = {cls.max():.3f} at {stall:.0f} deg")
    print(f"  min Cd (drag bucket)    = {cds.min():.3f}")
    print("=" * 60)
    sym_ok = abs(cl0) < 0.05
    slope_ok = 3.0 < slope_per_rad < 2 * np.pi + 0.3
    print(f"  symmetry (Cl0~0)            : {'OK' if sym_ok else 'CHECK'}")
    print(f"  lift slope plausible (3..2pi): {'OK' if slope_ok else 'CHECK'}")
    print("=" * 60)


def _plot(alphas, cls, cds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(alphas, cls, "o-", color="C0")
    ax[0].set_xlabel("angle of attack (deg)"); ax[0].set_ylabel("Cl")
    ax[0].set_title("lift curve"); ax[0].grid(alpha=0.3)
    ax[1].plot(alphas, cds, "o-", color="C3")
    ax[1].set_xlabel("angle of attack (deg)"); ax[1].set_ylabel("Cd")
    ax[1].set_title("drag bucket"); ax[1].grid(alpha=0.3)
    ax[2].plot(cds, cls, "o-", color="C2")
    ax[2].set_xlabel("Cd"); ax[2].set_ylabel("Cl")
    ax[2].set_title("drag polar"); ax[2].grid(alpha=0.3)
    fig.suptitle("NACA 0012 polar (2D LBM reference solver)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "airfoil_polar.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
