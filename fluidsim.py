"""
FluidSim -- a free, open-source GPU wind tunnel for the RC community.

One entry point for the whole tool. Import an .stl of your design and either
*watch* the air flow over it in real time, or *report* the aerodynamic numbers.
The desktop app has two modes: a wind tunnel (hold a model in the wind) and a
propeller test (spin a prop on the spot and read swirl / torque).

    python fluidsim.py gui                     # the desktop app (recommended)
    python fluidsim.py view  myplane.stl     # quick real-time viewer
    python fluidsim.py report myplane.stl     # run + print drag/forces
    python fluidsim.py demo                    # built-in wing, no file needed
    python fluidsim.py validate                # run the validation suite

Honest scope (see docs/VALIDATION.md): FluidSim excels at flow VISUALISATION and
at COMPARING designs (is A lower-drag than B, where does it stall). Its absolute
numbers are trustworthy for bluff bodies and attached flows (validated against
the cylinder, sphere and Schaefer-Turek benchmarks); they are NOT certified for
transition-dominated low-Reynolds airfoils. It tells you which is which.

Requires an NVIDIA GPU:  pip install -r requirements.txt -r requirements-gpu.txt
"""

from __future__ import annotations

import argparse
import os
import sys


def _need_gpu():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        pass
    sys.exit("No NVIDIA GPU / CuPy found. Install the GPU requirements:\n"
             "  pip install -r requirements-gpu.txt")


def cmd_gui(args):
    _need_gpu()
    if args.stl and not os.path.exists(args.stl):
        sys.exit(f"File not found: {args.stl}")
    from PySide6 import QtWidgets
    from fluidsim_gui import MainWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = MainWindow(args.stl)
    win.show()
    sys.exit(app.exec())


def cmd_view(args):
    _need_gpu()
    from live_viewer_3d import Viewer3D
    if args.stl and not os.path.exists(args.stl):
        sys.exit(f"File not found: {args.stl}")
    v = Viewer3D(args.n[0], args.n[1], args.n[2], args.aoa, stl_path=args.stl)
    v.run()


def cmd_demo(args):
    args.stl = None
    cmd_view(args)


def cmd_report(args):
    name = _need_gpu()
    import numpy as np
    import cupy as cp
    from lbm3d_cuda import LBM3D_CUDA
    from stl_import import load_binary_stl, fit_to_grid, voxelize
    if not os.path.exists(args.stl):
        sys.exit(f"File not found: {args.stl}")

    nx, ny, nz = args.n
    print(f"FluidSim report  |  {os.path.basename(args.stl)}  |  {name}")
    print(f"voxelising into {nx}x{ny}x{nz} ...")
    tris = load_binary_stl(args.stl)
    mask = voxelize(fit_to_grid(tris, nx, ny, nz, 0.22), nx, ny, nz)
    cells = int(mask.sum())
    if cells == 0:
        sys.exit("Empty voxelisation -- is this a valid binary STL?")

    sim = LBM3D_CUDA(nx, ny, nz, u_lb=0.05, re=args.re,
                     char_length=nx * 0.3, collision="les")
    sim.set_solid(mask)
    frontal = float(mask.any(axis=0).sum())          # area normal to the flow
    print(f"{cells} solid cells, frontal area {frontal:.0f} cells, Re={args.re:.0f}")
    print("running (LES) ...")
    steps = args.steps
    fx = []
    for t in range(steps):
        sim.step()
        if t > 0.6 * steps and t % 300 == 0:
            fx.append(sim.compute_force())
    cp.cuda.Stream.null.synchronize()
    F = np.mean(fx, axis=0)
    denom = 0.5 * sim.u_lb ** 2 * frontal
    print("\n" + "=" * 56)
    print("  AERODYNAMIC REPORT")
    print("=" * 56)
    print(f"  drag force (along flow)   Fx = {F[0]:+.4f}")
    print(f"  side / lift forces        Fy = {F[1]:+.4f}  Fz = {F[2]:+.4f}")
    print(f"  drag coefficient (frontal) Cd = {F[0]/denom:.3f}")
    print("=" * 56)
    print("  Use these for COMPARISON between designs and for trends.")
    print("  Absolute values are trustworthy for bluff/attached shapes;")
    print("  not certified for transitional low-Re airfoils. See VALIDATION.md")
    print("=" * 56)


def cmd_validate(args):
    _need_gpu()
    import subprocess
    suite = ["validate_cylinder.py", "validate_taylor_couette.py",
             "validate_sphere.py", "validate_prop.py", "gpu_benchmark.py"]
    print("Running the validation suite (this takes a few minutes):")
    for s in suite:
        print(f"\n--- {s} ---")
        subprocess.run([sys.executable, s])


def main():
    p = argparse.ArgumentParser(
        prog="fluidsim", description="A GPU wind tunnel for the RC community.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--n", type=int, nargs=3, default=[160, 100, 130],
                        metavar=("NX", "NY", "NZ"), help="grid resolution")
        sp.add_argument("--aoa", type=float, default=8.0, help="angle of attack")
        sp.add_argument("--re", type=float, default=600.0, help="Reynolds number")

    g = sub.add_parser("gui", help="the desktop application (recommended)")
    g.add_argument("stl", nargs="?", default=None)
    add_common(g); g.set_defaults(func=cmd_gui)

    v = sub.add_parser("view", help="quick real-time 3D viewer (no window chrome)")
    v.add_argument("stl", nargs="?", default=None)
    add_common(v); v.set_defaults(func=cmd_view)

    r = sub.add_parser("report", help="run and print aerodynamic numbers")
    r.add_argument("stl")
    r.add_argument("--steps", type=int, default=8000)
    add_common(r); r.set_defaults(func=cmd_report)

    d = sub.add_parser("demo", help="built-in wing showcase (no STL needed)")
    add_common(d); d.set_defaults(func=cmd_demo)

    val = sub.add_parser("validate", help="run the validation suite")
    val.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
