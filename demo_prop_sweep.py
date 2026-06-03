"""
demo_prop_sweep -- propeller performance map, headless.

Two modes:
  * hover  (default) -- the lift map for a rotor/drone prop: hold flight speed at
    static (J=0) and sweep RPM, reporting thrust (lift), shaft power, figure of
    merit and lift-per-power vs tip speed. This answers "how much lift, and how
    efficiently, across the throttle range" -- the useful question for a lifting
    prop (where cruise efficiency is 0 by definition).
  * cruise -- the classic airplane-prop map: hold RPM and sweep advance ratio J,
    reporting thrust & power and C_T / C_P / efficiency vs J.

Honesty: shaft power, torque and C_P are the robust (wall-carried angular
momentum) numbers. Thrust, figure of merit, C_T and efficiency carry the
documented low-Re voxel force caveat (see docs/VALIDATION.md) -- trustworthy for
comparing props and reading curve shapes, not as absolute certified values.

Usage:
    python demo_prop_sweep.py [model.stl] [--mode hover|cruise]
                              [--points N] [--settle S] [--rpm 0..1]
"""

from __future__ import annotations

import os
import sys
import argparse


def _need_gpu():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return
    except Exception:
        pass
    sys.exit("No NVIDIA GPU / CuPy found -- the prop solver is GPU-only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", nargs="?", default=None,
                    help="propeller STL (defaults to the built-in test prop)")
    ap.add_argument("--mode", choices=("hover", "cruise"), default="hover")
    ap.add_argument("--points", type=int, default=7, help="sample count")
    ap.add_argument("--settle", type=int, default=4000, help="steps per point")
    ap.add_argument("--rpm", type=float, default=0.6,
                    help="cruise mode: RPM fraction 0..1 held fixed")
    args = ap.parse_args()
    if args.stl and not os.path.exists(args.stl):
        sys.exit(f"File not found: {args.stl}")
    _need_gpu()

    from prop_model import PropModel
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)

    print("Building & warming up the prop …", flush=True)
    model = PropModel()
    cells = model.load(args.stl)                 # auto-orients any prop STL
    print(f"  {cells} solid cells", flush=True)

    if args.mode == "hover":
        def prog(i, n, m):
            print(f"  point {i}/{n}:  tipMach={m['tip_mach']:.2f}  "
                  f"thrust={m['thrust']:+.4f}  power={abs(m['power']):.4f}  "
                  f"FM={100*m['fom']:4.0f}%  lift/pwr={m['lift_per_power']:.1f}",
                  flush=True)
        print(f"\nHover lift sweep: static, {args.points} RPM points "
              f"({args.settle} steps each) …", flush=True)
        res = model.sweep_rpm(n_points=args.points, settle=args.settle, progress=prog)
        ceil = res.get("diverged_at_tip_mach")
        if ceil is not None:
            print(f"  [note] flow diverged at tip Mach ~{ceil:.2f} -- near-static "
                  "is only stable at low RPM in this incompressible solver; "
                  f"reporting the {len(res['tip_mach'])} stable point(s).", flush=True)
        png = _plot_hover(res, out)
        head = "PROPELLER HOVER LIFT MAP  (near-static)"
        robust, est = "shaft power", "thrust, figure of merit, lift-per-power"
    else:
        model.set_rpm_fraction(args.rpm)         # fix RPM for the J sweep
        def prog(i, n, m):
            print(f"  point {i}/{n}:  J={m['advance_j']:5.2f}  "
                  f"thrust={m['thrust']:+.4f}  power={abs(m['power']):.4f}  "
                  f"C_T={m['c_t']:+.3f}  C_P={m['c_p']:.3f}  eta={100*m['eta']:4.0f}%",
                  flush=True)
        print(f"\nCruise sweep: RPM {args.rpm}, {args.points} J points "
              f"({args.settle} steps each) …", flush=True)
        res = model.sweep_j(n_points=args.points, settle=args.settle, progress=prog)
        png = _plot_cruise(res, out)
        head = "PROPELLER CRUISE MAP  (fixed RPM)"
        robust, est = "shaft power, C_P", "thrust, C_T, efficiency"

    print("\n" + "=" * 64)
    print(f"  {head}")
    print("=" * 64)
    print(f"  robust    : {robust}")
    print(f"  estimated : {est}  (low-Re voxel caveat)")
    print(f"  curve saved -> {png}")
    print("=" * 64)


def _plot_hover(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = res["tip_mach"]
    if not x:
        return "(no stable points — hover diverged at the lowest RPM)"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, res["thrust"], "s-", color="#e76f51", label="thrust / lift (est.)")
    ax[0].plot(x, res["power"], "o-", color="#2a9d8f", label="shaft power (robust)")
    ax[0].set_xlabel("tip Mach  (RPM)"); ax[0].set_ylabel("lattice units")
    ax[0].set_title("Lift & power vs RPM"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, [100.0 * f for f in res["fom"]], "^-", color="#264653",
               label="figure of merit % (est.)")
    ax[1].plot(x, res["lift_per_power"], "d--", color="#e76f51",
               label="lift ÷ power (est.)")
    ax[1].set_xlabel("tip Mach  (RPM)")
    ax[1].set_title("Lift efficiency vs RPM"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("Hover lift map (static, J=0) — "
                 "robust: power · estimated: thrust / FM / lift-per-power")
    fig.tight_layout()
    path = os.path.join(out, "prop_hover_map.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def _plot_cruise(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = res["j"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(j, res["power"], "o-", color="#2a9d8f", label="shaft power (robust)")
    ax[0].plot(j, res["thrust"], "s--", color="#e76f51", label="thrust (est.)")
    ax[0].set_xlabel("advance ratio  J"); ax[0].set_ylabel("lattice units")
    ax[0].set_title("Thrust & power vs J"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(j, res["c_p"], "o-", color="#2a9d8f", label="C_P (robust)")
    ax[1].plot(j, res["c_t"], "s--", color="#e76f51", label="C_T (est.)")
    ax[1].plot(j, [100.0 * e for e in res["eta"]], "^:", color="#264653",
               label="η % (est.)")
    ax[1].set_xlabel("advance ratio  J")
    ax[1].set_title("Coefficients & efficiency vs J")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("Propeller cruise map (fixed RPM) — "
                 "robust: power/C_P · estimated: thrust/C_T/η")
    fig.tight_layout()
    path = os.path.join(out, "prop_performance.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


if __name__ == "__main__":
    main()
