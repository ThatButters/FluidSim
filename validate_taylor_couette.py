"""
validate_taylor_couette -- moving-boundary go/no-go gate.

Taylor-Couette flow: a rotating inner cylinder (radius r1, angular velocity
omega) inside a stationary outer cylinder (radius r2). The steady azimuthal
velocity has an EXACT analytical solution:

    u_theta(r) = A r + B / r,   A = omega r1^2 / (r1^2 - r2^2),  B = -A r2^2

If our moving-wall bounce-back reproduces this profile, the core rotating-
boundary physics is correct -- the make-or-break feature of the whole project.

Closed domain (no through-flow): the outer solid region fills every domain-edge
cell, so the inherited inlet/outlet conditions act only on solid cells and are
harmless.
"""

from __future__ import annotations

import os
import numpy as np

from lbm2d_rotating import RotatingLBM2D

OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    nx = ny = 200
    cx = cy = 100.0
    r1, r2 = 25.0, 85.0
    nu = 0.10                                # fast momentum diffusion
    u_wall = 0.04                            # inner-wall speed (low Mach)
    omega = u_wall / r1

    sim = RotatingLBM2D(nx, ny, u_lb=u_wall, re=100.0, char_length=2 * r1)
    sim.omega = 1.0 / (3.0 * nu + 0.5)       # set viscosity directly
    sim.vel[:] = 0.0                         # quiescent start, no through-flow
    sim.f = sim._equilibrium(np.ones((nx, ny)), np.zeros((2, nx, ny)))

    x = np.arange(nx)[:, None] - cx
    y = np.arange(ny)[None, :] - cy
    rr = np.sqrt(x ** 2 + y ** 2)
    inner = rr < r1
    outer = rr > r2
    sim.solid = inner | outer
    sim.body = inner                         # measure torque-bearing inner wall
    sim.set_rotation(cx, cy, omega, inner)

    print(f"Taylor-Couette  {nx}x{ny}  r1={r1:.0f} r2={r2:.0f}  "
          f"omega={omega:.5f}  nu={nu}  lbm_omega={sim.omega:.3f}", flush=True)

    steps = 80000
    prev = None
    for t in range(steps):
        sim.step()
        if t % 8000 == 0:
            _, u = sim.macroscopic()
            speed = np.sqrt(u[0] ** 2 + u[1] ** 2)
            fluid = ~sim.solid
            m = float(speed[fluid].mean())
            conv = "" if prev is None else f"  d={abs(m-prev):.2e}"
            print(f"  step {t:6d}/{steps}  mean|u|={m:.5f}{conv}", flush=True)
            prev = m

    _analyse(sim, cx, cy, r1, r2, omega, nu, u_wall)


def _analyse(sim, cx, cy, r1, r2, omega, nu, u_wall):
    nx, ny = sim.nx, sim.ny
    _, u = sim.macroscopic()
    x = np.arange(nx)[:, None] - cx
    y = np.arange(ny)[None, :] - cy
    r = np.sqrt(x ** 2 + y ** 2)
    # Azimuthal (tangential) velocity component: u_theta = u . t_hat,
    # t_hat = (-sin, cos) = (-y, x)/r.
    with np.errstate(invalid="ignore", divide="ignore"):
        u_theta = (-y * u[0] + x * u[1]) / r

    # Analytical profile.
    A = omega * r1 ** 2 / (r1 ** 2 - r2 ** 2)
    B = -A * r2 ** 2

    # Bin measured u_theta by radius over the interior of the annulus.
    radii = np.arange(int(r1) + 3, int(r2) - 2)
    meas, exact = [], []
    fluid = ~sim.solid
    for rad in radii:
        ring = fluid & (r >= rad - 0.5) & (r < rad + 0.5)
        if ring.sum() < 8:
            continue
        meas.append(float(u_theta[ring].mean()))
        exact.append(A * rad + B / rad)
    meas, exact = np.array(meas), np.array(exact)
    rel = np.abs(meas - exact) / np.max(np.abs(exact))
    rms = float(np.sqrt(np.mean(rel ** 2)))
    mx = float(rel.max())

    np.savez(os.path.join(OUT, "taylor_couette.npz"),
             radii=radii[:meas.size], meas=meas, exact=exact)
    _plot(radii[:meas.size], meas, exact)

    ok = rms < 0.05
    print("\n" + "=" * 58)
    print("  TAYLOR-COUETTE (moving-boundary) VALIDATION")
    print("=" * 58)
    print(f"  RMS rel. error vs exact u_theta(r) = {rms:.3%}")
    print(f"  max rel. error                     = {mx:.3%}")
    print(f"  sampled {meas.size} radial stations across the annulus")
    print("=" * 58)
    print(f"  {'GO -- moving-boundary physics validated.' if ok else 'investigate: profile off >5%.'}")
    print("=" * 58)


def _plot(radii, meas, exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(radii, exact, "k-", lw=2, label="exact  A r + B/r")
    plt.plot(radii, meas, "o", ms=4, color="C3", label="LBM (moving wall)")
    plt.xlabel("radius r (lattice units)")
    plt.ylabel(r"azimuthal velocity $u_\theta$")
    plt.title("Taylor-Couette: rotating-boundary LBM vs analytical")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "taylor_couette.png"), dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
