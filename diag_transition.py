"""
diag_transition -- does near-DNS resolution recover the E387 lift?

The decisive test of the transition diagnosis. We hold the airfoil at alpha=4
(UIUC reference Cl ~ 0.80) and push the resolution up. If the laminar separation
bubble's transition/reattachment is what we're missing, lift should eventually
TURN AROUND and climb toward the data as the grid resolves it. If lift keeps
sliding, the gap is something deeper than transition-by-resolution.

Each run is taken to convergence (lift plateau) and the Cl trace is printed so we
can see it has settled, not just sampled mid-transient.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from validate_e387 import load_airfoil, airfoil_mask
from lbm2d_cuda import LBM2D_CUDA

U = 0.05
RE = 100000.0
AOA = 4.0
REF_CL = 0.80


def run(coords, chord):
    nx = int(round(2.6 * chord / 2) * 2)
    ny = int(round(1.8 * chord / 2) * 2)
    steps = int(5 * chord / U)
    sim = LBM2D_CUDA(nx, ny, u_lb=U, re=RE, char_length=chord, collision="les")
    sim.set_solid(airfoil_mask(coords, nx, ny, chord, 1.1 * chord, ny / 2.0, AOA))
    cells = nx * ny
    print(f"  chord={chord:5.0f}  {nx}x{ny} ({cells/1e6:4.1f}M)  steps={steps}",
          flush=True)
    trace = []
    for t in range(steps):
        sim.step()
        if t % (steps // 5) == 0 and t > 0:
            cp.cuda.Stream.null.synchronize()
            trace.append(sim.coefficients()[1])
    cp.cuda.Stream.null.synchronize()
    cl = float(sim.coefficients()[1])
    finite = bool(cp.all(cp.isfinite(sim.f_a)))
    print(f"           Cl trace: {['%.3f' % c for c in trace]}  -> {cl:.3f}"
          f"  finite={finite}", flush=True)
    return cl


def main():
    coords = load_airfoil()
    print(f"E387 alpha={AOA:.0f}, Re={RE:.0f}.  UIUC reference Cl ~ {REF_CL}")
    print("Resolution trend (does lift turn the corner toward the data?):")
    chords = [600, 1200, 1800, 2400]
    cls = [run(coords, c) for c in chords]
    print("\n" + "=" * 52)
    print("  RESOLUTION TREND")
    print("=" * 52)
    for c, cl in zip(chords, cls):
        print(f"  chord {c:5d}   Cl = {cl:.3f}   (ref {REF_CL})")
    turned = cls[-1] > cls[1] + 0.02
    print("=" * 52)
    print(f"  lift {'CLIMBS with resolution -> transition recovery likely' if turned else 'does NOT recover -> deeper than transition-by-resolution'}")
    print("=" * 52)


if __name__ == "__main__":
    main()
