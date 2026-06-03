"""
validate_sphere -- 3D solver validation: flow past a sphere at Re=100.

The 3D counterpart of the 2D cylinder gate. A sphere at Re=100 has a steady,
axisymmetric wake (no shedding below Re~210), so the drag converges to a single
value with a well-known reference:

    Schiller-Naumann:  Cd = (24/Re)(1 + 0.15 Re^0.687)  ->  ~1.09 at Re=100

Cd = Fx / (0.5 rho U^2 * pi R^2)  (frontal area).

As with the first 2D cylinder run, a CPU/GPU-affordable domain has non-trivial
blockage and modest sphere resolution, both of which inflate Cd above the
free-stream reference -- reported honestly. The point is that the 3D solver
produces the right physics and a drag in the right range.
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

OUT = os.path.join(os.path.dirname(__file__), "out")


def schiller_naumann(re):
    return (24.0 / re) * (1.0 + 0.15 * re ** 0.687)


def main():
    nx, ny, nz = 200, 120, 120
    r = 12.0
    D = 2 * r
    re = 100.0
    sim = LBM3D(nx, ny, nz, u_lb=0.05, re=re, char_length=D,
                array_module=XP, dtype=np.float32 if XP is np else cp.float32)
    sim.add_sphere(cx=50.0, cy=ny / 2.0, cz=nz / 2.0, r=r)
    area = np.pi * r ** 2
    blockage = (np.pi * r ** 2) / (ny * nz)
    backend = "GPU" if XP is not np else "CPU"
    print(f"Sphere Re={re:.0f} on {backend}  {nx}x{ny}x{nz} "
          f"({nx*ny*nz/1e6:.1f}M cells)  D={D:.0f}  "
          f"area-blockage={blockage:.1%}  omega={sim.omega:.3f}", flush=True)

    steps = 14000
    cd_hist = np.empty(steps)
    for t in range(steps):
        sim.step()
        cd_hist[t] = sim.drag_coefficient(area)
        if t % 1000 == 0:
            if XP is not np:
                cp.cuda.Stream.null.synchronize()
            print(f"  step {t:6d}/{steps}  Cd={cd_hist[t]:+.3f}", flush=True)

    if not np.all(np.isfinite(cd_hist)):
        raise RuntimeError("Sphere run diverged.")

    cd = float(cd_hist[-2000:].mean())
    drift = float(abs(cd_hist[-1] - cd_hist[-2000]))
    ref = schiller_naumann(re)
    _slice_plot(sim)

    print("\n" + "=" * 60)
    print("  SPHERE DRAG (3D D3Q19) VALIDATION")
    print("=" * 60)
    print(f"  converged Cd            = {cd:.3f}  (drift over last 2k = {drift:.3f})")
    print(f"  Schiller-Naumann ref    = {ref:.3f}  (free-stream)")
    print(f"  ratio Cd/ref            = {cd/ref:.2f}  "
          f"(>1 expected: blockage {blockage:.0%} + D={D:.0f} resolution)")
    print("=" * 60)
    # Steady, finite, positive, and within a blockage/resolution-explained band.
    ok = np.isfinite(cd) and 0.9 < cd < 1.8 and drift < 0.05
    print(f"  {'GO -- 3D solver produces correct sphere physics.' if ok else 'review.'}")
    print("=" * 60)


def _slice_plot(sim):
    """Save a mid-plane (z = nz/2) speed slice through the 3D field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho, u = sim.macroscopic()
    k = sim.nz // 2
    sp = np.sqrt(np.asarray((u[0] ** 2 + u[1] ** 2 + u[2] ** 2)[:, :, k]
                            if XP is np else
                            cp.asnumpy((u[0] ** 2 + u[1] ** 2 + u[2] ** 2)[:, :, k])))
    solid = sim.solid[:, :, k]
    solid = solid if XP is np else cp.asnumpy(solid)
    sp[solid] = np.nan
    plt.figure(figsize=(10, 6))
    plt.imshow(sp.T, origin="lower", cmap="viridis")
    plt.title("Flow past a sphere (Re=100) -- speed, mid-plane slice")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "sphere_slice.png"), dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
