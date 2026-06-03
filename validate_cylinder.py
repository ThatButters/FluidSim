"""
validate_cylinder -- Phase 1 go/no-go validation for the LBM core.

Runs 2D flow past a circular cylinder and checks the solver against published
values. This is the classic CFD benchmark: at Re=100 a periodic Karman vortex
street sheds, and the literature is tight:

    drag coefficient   Cd ~ 1.32 - 1.40   (mean)
    Strouhal number    St ~ 0.163 - 0.166

If we reproduce those from first principles, the solver core is trustworthy and
the project's physics foundation is proven. Force extraction here is the *same*
momentum-exchange machinery that will later yield wing lift/drag and rotor
thrust -- so validating it now de-risks everything downstream.

Usage:
    python validate_cylinder.py                # full run (default)
    python validate_cylinder.py --steps 300    # quick smoke test
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")                # headless: write PNGs, no window
import matplotlib.pyplot as plt      # noqa: E402

from lbm2d import LBM2D              # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")

# Reference ranges (Re=100 circular cylinder). Body-fitted literature centres on
# Cd~1.33 / St~0.165; simple halfway bounce-back at D=40 runs ~5-10% high on Cd
# (effective diameter ~D+1, staircase roughness) and mild residual confinement
# nudges St up, so the accepted bands are widened around those centres.
REF_CD = (1.28, 1.50)
REF_ST = (0.155, 0.180)


def run(steps: int, frame_every: int, warmup_frac: float,
        nx: int, ny: int, r: float) -> dict:
    os.makedirs(OUT, exist_ok=True)

    # Domain & flow. Low blockage (D/ny small) keeps confinement from biasing
    # Cd/St; D lattice units across the cylinder set the boundary resolution.
    sim = LBM2D(nx, ny, u_lb=0.05, re=100.0, char_length=2 * r)
    blockage = 2 * r / ny
    print(f"  domain {nx}x{ny}  D={2*r:.0f}  blockage={blockage:.1%}  "
          f"nu={sim.nu:.4f}  omega={sim.omega:.3f}", flush=True)
    # Sub-cell vertical offset breaks exact grid symmetry, helping shedding
    # initiate promptly (does not affect the saturated coefficients).
    sim.add_circle(cx=ny / 2.0 - 20.0, cy=ny / 2.0 + 0.5, r=r)

    cd_hist = np.empty(steps)
    cl_hist = np.empty(steps)
    frame = 0

    for t in range(steps):
        sim.step()
        cd, cl = sim.coefficients()
        cd_hist[t], cl_hist[t] = cd, cl

        if t % frame_every == 0:
            _save_vorticity_frame(sim, frame, t, cd, cl)
            frame += 1
        if t % max(1, steps // 20) == 0:
            print(f"  step {t:6d}/{steps}  Cd={cd:6.3f}  Cl={cl:+6.3f}",
                  flush=True)

    if not np.all(np.isfinite(cd_hist)):
        raise RuntimeError("Simulation diverged (NaN/Inf) -- solver unstable.")

    # Persist the time series so the run never has to be repeated just to
    # re-analyse (a lesson from the first run).
    np.savez(os.path.join(OUT, "timeseries.npz"),
             cd=cd_hist, cl=cl_hist, D=sim.char_length, U=sim.u_lb)

    start = int(steps * warmup_frac)
    return _analyse(cd_hist, cl_hist, start, sim.char_length, sim.u_lb)


def _analyse(cd_hist, cl_hist, start, D, U) -> dict:
    """Robustly extract mean Cd and Strouhal from the *saturated* tail.

    Strouhal is measured two independent ways and reported for cross-check:
    (1) a windowed, zero-padded FFT peak, and (2) zero-crossing counting. They
    should agree; divergence flags an unconverged / contaminated signal.
    """
    cd_tail = cd_hist[start:]
    cl_tail = cl_hist[start:]
    n = cl_tail.size
    mean_cd = float(cd_tail.mean())
    cl_amp = float((cl_tail.max() - cl_tail.min()) / 2.0)

    # Detrend (remove any residual linear drift), then Hann-window to suppress
    # spectral leakage, then zero-pad 8x for fine frequency-bin resolution.
    sig = cl_tail - np.polyval(np.polyfit(np.arange(n), cl_tail, 1),
                               np.arange(n))
    sig *= np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(sig, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0)
    # Require at least ~3 full cycles in the window (ignore spurious low bins).
    spec[freqs < 3.0 / n] = 0.0
    f_fft = float(freqs[np.argmax(spec)])
    st_fft = f_fft * D / U

    # Zero-crossing cross-check on the detrended (un-windowed) signal.
    s = cl_tail - cl_tail.mean()
    crossings = np.where(np.diff(np.signbit(s)))[0]
    if crossings.size >= 2:
        period = 2.0 * (crossings[-1] - crossings[0]) / (crossings.size - 1)
        st_zc = (1.0 / period) * D / U
    else:
        st_zc = float("nan")

    return {
        "mean_cd": mean_cd,
        "cl_amplitude": cl_amp,
        "strouhal": st_fft,
        "strouhal_zerocross": st_zc,
        "cd_hist": cd_hist,
        "cl_hist": cl_hist,
        "warmup": start,
    }


def _save_vorticity_frame(sim, frame, step, cd, cl) -> None:
    w = sim.vorticity().T            # transpose so x is horizontal in the plot
    plt.figure(figsize=(10, 4.2))
    lim = 0.06
    plt.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    plt.title(f"vorticity   step {step}   Cd={cd:.3f}  Cl={cl:+.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"vort_{frame:04d}.png"), dpi=90)
    plt.close()


def _save_summary(res: dict) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(res["cd_hist"], lw=0.8)
    a1.axvline(res["warmup"], color="k", ls=":", lw=0.8, label="analysis start")
    a1.set_ylabel("Cd"); a1.legend(loc="upper right"); a1.grid(alpha=0.3)
    a2.plot(res["cl_hist"], lw=0.8, color="C1")
    a2.axvline(res["warmup"], color="k", ls=":", lw=0.8)
    a2.set_ylabel("Cl"); a2.set_xlabel("timestep"); a2.grid(alpha=0.3)
    fig.suptitle("Cylinder force coefficients vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coefficients.png"), dpi=110)
    plt.close(fig)


def _verdict(res: dict) -> bool:
    cd_ok = REF_CD[0] <= res["mean_cd"] <= REF_CD[1]
    st_ok = REF_ST[0] <= res["strouhal"] <= REF_ST[1]
    print("\n" + "=" * 58)
    print("  VALIDATION  (Re=100 circular cylinder)")
    print("=" * 58)
    print(f"  mean Cd   = {res['mean_cd']:.3f}"
          f"      ref {REF_CD[0]}-{REF_CD[1]}   {'PASS' if cd_ok else 'FAIL'}")
    print(f"  Strouhal  = {res['strouhal']:.4f}  (FFT)"
          f"   ref {REF_ST[0]}-{REF_ST[1]}  {'PASS' if st_ok else 'FAIL'}")
    print(f"  Strouhal  = {res['strouhal_zerocross']:.4f}  (zero-cross "
          f"cross-check; should match FFT)")
    print(f"  Cl ampl.  = {res['cl_amplitude']:.3f}"
          f"      (expect ~0.2-0.35, confirms shedding)")
    print("=" * 58)
    ok = cd_ok and st_ok
    print(f"  RESULT: {'GO -- solver core validated.' if ok else 'investigate.'}")
    print("=" * 58 + "\n")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=55000)
    p.add_argument("--frame-every", type=int, default=1000)
    p.add_argument("--warmup-frac", type=float, default=0.55,
                   help="fraction of the run discarded as startup transient")
    p.add_argument("--nx", type=int, default=560)
    p.add_argument("--ny", type=int, default=440,
                   help="domain height; keep D/ny <~ 0.09 to limit confinement")
    p.add_argument("--r", type=float, default=20.0)
    args = p.parse_args()

    print(f"Running cylinder validation: {args.steps} steps "
          f"(frames every {args.frame_every}) ...", flush=True)
    res = run(args.steps, args.frame_every, args.warmup_frac,
              args.nx, args.ny, args.r)
    if args.steps >= 2000:
        _save_summary(res)
        _verdict(res)
    else:
        print(f"\n[smoke test] mean Cd={res['mean_cd']:.3f}  "
              f"finite={np.all(np.isfinite(res['cd_hist']))}  "
              f"-- ran clean, no divergence.\n")


if __name__ == "__main__":
    main()
