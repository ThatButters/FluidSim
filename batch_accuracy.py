"""
batch_accuracy -- the high-accuracy (near-DNS) batch mode.

The companion to the real-time interactive mode. Where the live viewer trades
accuracy for speed (coarse grid, watch it now), this trades speed for accuracy:
run a high-resolution case offline, time-average the unsteady loads over many
shedding cycles, and report a converged coefficient with its grid-convergence
trend. Minutes, not milliseconds.

SCOPE (honest): this is trustworthy for the regimes the solver is validated in --
bluff bodies and attached / geometry-fixed-separation flows (cylinder, sphere,
DFG all pass). It is NOT trustworthy for transition-dominated low-Re airfoils:
2D simulation spuriously stalls them (it cannot represent the 3D transition that
keeps the real boundary layer attached), so the time-averaged lift is grid-
converged but well below the wind-tunnel value. See docs/VALIDATION.md.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import cupy as cp

from lbm2d_cuda import LBM2D_CUDA
from validate_e387 import load_airfoil, airfoil_mask

OUT = os.path.join(os.path.dirname(__file__), "out")


def run_case(coords, re, aoa, chord, u_lb=0.05, develop_frac=2.0,
             average_frac=1.6, sample=400, progress=False):
    """One high-accuracy case: develop, then time-average Cl/Cd over the
    unsteady tail. Returns (Cl_mean, Cl_std, Cd_mean, Cd_std)."""
    nx = int(round(2.6 * chord / 2) * 2)
    ny = int(round(1.8 * chord / 2) * 2)
    develop = int(develop_frac * nx / u_lb)
    average = int(average_frac * nx / u_lb)
    sim = LBM2D_CUDA(nx, ny, u_lb=u_lb, re=re, char_length=chord,
                     collision="les")
    sim.set_solid(airfoil_mask(coords, nx, ny, chord, 1.1 * chord,
                               ny / 2.0, aoa))
    if progress:
        print(f"    [{nx}x{ny}={nx*ny/1e6:.0f}M cells, develop {develop} + "
              f"average {average} steps]", flush=True)
    for _ in range(develop):                      # let the wake establish
        sim.step()
    cl, cd = [], []
    for t in range(average):                      # time-average the unsteady tail
        sim.step()
        if t % sample == 0:
            d, l = sim.coefficients()
            cl.append(l); cd.append(d)
    cp.cuda.Stream.null.synchronize()
    cl, cd = np.array(cl), np.array(cd)
    if not (np.all(np.isfinite(cl)) and np.all(np.isfinite(cd))):
        return (float("nan"),) * 4
    return cl.mean(), cl.std(), cd.mean(), cd.std()


def high_accuracy(coords, re, aoa, chords=(2000, 3000), **kw):
    """Run at two resolutions and report the grid-convergence trend."""
    print(f"  AoA={aoa:+.0f}  Re={re:.0f}:", flush=True)
    res = []
    for c in chords:
        clm, cls, cdm, cds = run_case(coords, re, aoa, c, progress=True, **kw)
        res.append((c, clm, cls, cdm, cds))
        print(f"    chord {c:5d}:  Cl = {clm:.3f} +/- {cls:.3f}   "
              f"Cd = {cdm:.4f} +/- {cds:.4f}", flush=True)
    # grid-convergence read: change between the two finest resolutions.
    dcl = abs(res[-1][1] - res[-2][1])
    converged = dcl < 0.05
    print(f"    -> Cl change over last refinement: {dcl:.3f}  "
          f"({'grid-converged' if converged else 'still refining'})")
    return res


def main():
    p = argparse.ArgumentParser(description="High-accuracy batch CFD")
    p.add_argument("--re", type=float, default=100000.0)
    p.add_argument("--aoa", type=float, default=4.0)
    p.add_argument("--chords", type=int, nargs="+", default=[2000, 3000])
    args = p.parse_args()

    coords = load_airfoil()
    print("=" * 60)
    print("  HIGH-ACCURACY BATCH MODE  (near-DNS + time-averaging)")
    print("=" * 60)
    print("  E387 airfoil vs UIUC wind-tunnel reference")
    res = high_accuracy(coords, args.re, args.aoa, tuple(args.chords))
    ref = 0.40 + 0.10 * args.aoa                  # UIUC E387 linear reference
    best = res[-1][1]
    print("=" * 60)
    print(f"  best Cl = {best:.3f}   UIUC ref ~ {ref:.2f}   "
          f"({(best-ref)/ref*100:+.0f}% vs data)")
    print("  (vs the coarse interactive estimate ~0.40 at this angle)")
    print("=" * 60)


if __name__ == "__main__":
    main()
