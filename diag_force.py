"""
diag_force -- decide the correct momentum-exchange force formula empirically,
in a TRUSTED low-Mach regime (U=0.05, Mach~0.087).

The cylinder run reproduces the right Strouhal number (flow dynamics correct)
but under-predicts Cd with the current formula. From first principles, halfway
bounce-back delivers 2*c_i*f_i of momentum to the wall per boundary link, so the
rigorous drag is F = +sum 2 c_i f_i (post-collision). We compare:

    A (current): F = -sum_links c_i ( f_i + f_opp )    -> known to give ~1.0
    B (rigorous):F = +sum_links 2 c_i f_i              -> hypothesis: ~1.33

Whichever lands on the published free-stream Cd ~ 1.33 (Re=100) is correct.
Cl is tracked to confirm the flow is genuinely shedding (steady Cd + oscillating
Cl), i.e. the trustworthy wake, not a degenerate high-Mach one.
"""

from __future__ import annotations

import numpy as np

from lbm2d import LBM2D, C, OPP


def _links(solid):
    for i in range(9):
        if i == 4:
            continue
        neigh_solid = np.roll(np.roll(solid, -C[i, 0], axis=0),
                              -C[i, 1], axis=1)
        link = neigh_solid & ~solid
        if link.any():
            yield i, link


def cd_formula_A(fout, solid, denom):
    f = np.zeros(2)
    for i, link in _links(solid):
        f += C[i] * (fout[i][link] + fout[OPP[i]][link]).sum()
    return (-f)[0] / denom


def cd_formula_B(fout, solid, denom):
    f = np.zeros(2)
    for i, link in _links(solid):
        f += C[i] * (2.0 * fout[i][link].sum())
    return f[0] / denom            # positive sign (momentum into wall = drag)


def main():
    nx, ny, r = 480, 300, 20.0
    U = 0.05                                   # Mach ~0.087, trustworthy regime
    sim = LBM2D(nx, ny, u_lb=U, re=100.0, char_length=2 * r)
    sim.add_circle(cx=ny / 2.0 - 20.0, cy=ny / 2.0 + 0.5, r=r)
    denom = 0.5 * U ** 2 * (2 * r)

    steps = 40000
    start = int(steps * 0.6)
    cdA, cdB, cl = [], [], []
    print(f"blockage={2*r/ny:.1%}  omega={sim.omega:.3f}  Mach={U/0.5773:.3f}  "
          f"steps={steps}", flush=True)
    for t in range(steps):
        sim.step()
        if t >= start:
            cdA.append(cd_formula_A(sim._fout, sim.solid, denom))
            cdB.append(cd_formula_B(sim._fout, sim.solid, denom))
            cl.append(sim.coefficients()[1])
        if t % 4000 == 0:
            a = cd_formula_A(sim._fout, sim.solid, denom)
            b = cd_formula_B(sim._fout, sim.solid, denom)
            print(f"  step {t:6d}  Cd_A={a:6.3f}  Cd_B={b:6.3f}  "
                  f"Cl={sim.coefficients()[1]:+6.3f}", flush=True)

    cl = np.array(cl)
    print("\n" + "=" * 56)
    print(f"  mean Cd_A (-(f_i+f_opp)) = {np.mean(cdA):.3f}")
    print(f"  mean Cd_B (+2 f_i)       = {np.mean(cdB):.3f}")
    print(f"  published free-stream    ~ 1.33  (Re=100)")
    print(f"  Cl oscillation amplitude = {(cl.max()-cl.min())/2:.3f}  "
          f"(confirms shedding if ~0.2-0.3)")
    print("=" * 56)


if __name__ == "__main__":
    main()
