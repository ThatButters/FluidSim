# Validation & Accuracy

This is an honest record of what the solver has been checked against, how close
it gets, and — just as important — **what has not yet been validated.** Numbers
without this context are misleading, so we keep both in one place.

## What "validated" means here

Our checks fall into three tiers, and they are *not* equally strong evidence of
real-world accuracy:

1. **Exact mathematics** — a closed-form analytical solution. Strongest proof of
   *correctness*, but an idealised flow.
2. **Gold-standard numerical benchmark** — community-reference values agreed by
   many high-accuracy codes and consistent with experiment.
3. **Direct experiment** — measured in a real wind tunnel.

## Results so far (2D reference solver)

| Case | Reference | Tier | Result | Error |
| --- | --- | --- | --- | --- |
| Taylor–Couette (rotating cylinder) | exact $u_\theta(r)=Ar+B/r$ | 1 — exact math | velocity profile | **0.29 % RMS**, 0.51 % max |
| Schäfer–Turek 2D-2 (DFG cylinder) | Schäfer & Turek 1996 | 2 — benchmark | $C_{d,\max}=3.28$ | +1.5 % vs 3.22–3.24 |
| Schäfer–Turek 2D-2 | Schäfer & Turek 1996 | 2 — benchmark | $\mathrm{St}=0.306$ | +1 % vs 0.295–0.305 |
| Schäfer–Turek 2D-2 | Schäfer & Turek 1996 | 2 — benchmark | $C_{l,\max}=0.94$ | −6 % vs 0.99–1.01 |
| Cylinder vortex shedding (Re=100) | Williamson (experiment) | 3 — experiment | $\mathrm{St}=0.177$ | +7 % vs ~0.165 |
| Spinning rotor (sweeping geometry) | angular-momentum budget | self-consistency | stable, correct spin-up | — |
| NACA 0012 polar (Re=1000) | thin-airfoil theory + airfoil behaviour | 1/2 — theory | $C_{l,0}=0$, drag bucket, stall ✓; slope 2.3/rad | qualitatively correct; slope low (low-Re + resolution) |
| Sphere drag, 3D (Re=100) | Schiller–Naumann correlation | 2 — correlation | $C_d=1.21$ vs 1.09 | +11 % (D=24 staircasing; 3% blockage); steady wake correct |
| STL voxelisation | analytic sphere mask | round-trip | IoU 0.93 | mesh→voxels correct (faceting/rounding gap) |
| 3D CUDA force | CuPy reference (sphere) | cross-check | $C_d=1.205$ vs 1.208 | fused-kernel force matches reference to 0.25 % |
| 3D finite-wing polar (Re=600) | finite-wing theory | 1 — theory | slope 1.42/rad vs ~1.58 | tip-vortex lift loss + induced drag correct (within ~10%) |

**Reading the numbers.** Against experiment-validated references we are landing
within **roughly 1–7 %**:

- **Frequencies** (Strouhal / shedding rate) are excellent — ~1 % on the DFG
  benchmark. The 7 % on the free cylinder is **confinement**, not the solver: the
  CPU domain was not large enough to be truly "open," which raises Strouhal. A
  bigger grid tightens it toward ~1 %.
- **Integrated forces** (drag, lift) are within ~1.5 % (drag) to ~6 % (lift). Lift
  is the most resolution-sensitive and improves on finer grids.
- **Velocity fields** are sub-1 % — but that comparison is against exact math.

For a from-scratch solver at modest (CPU-affordable) resolution, this is solid.

## What has NOT been validated yet — and it's the part that matters most

**Every result above is at Reynolds number ≤ 100, in 2D, fully laminar.**

Real RC aircraft, helicopters, and drones operate at **Re ≈ 10⁴–10⁵**, in **3D**,
with **turbulence and laminar-to-turbulent transition** — a fundamentally harder
regime that determines real drag, stall, and thrust. **We have no accuracy number
there yet, and we will not claim one until we measure it.**

**The BGK stability ceiling, measured.** On the 3D wing we found plain BGK stays
stable to about **Re≈800** at this resolution (relaxation time $\tau\approx0.51$)
and **diverges by Re≈1200** ($\tau\to0.5$). So the finite-wing polar runs at
Re=600: the *physics* (finite-wing lift slope, induced drag) is correct, but the
*magnitude* is at low Re. Reaching RC Reynolds numbers (10⁴–10⁵) at affordable
resolution is precisely what a cumulant / regularised collision operator is for.

This is the known hard frontier (see the research that informed the project):
trustworthy accuracy at the RC operating regime requires

- the **GPU port**, for resolution we cannot afford on a CPU;
- a **better collision operator** than plain BGK (e.g. cumulant / regularised),
  for stability at high Reynolds number on affordable grids;
- and validation against **real RC datasets**, not just canonical flows.

The realistic ceiling reported in the literature is **~1 % on integrated
quantities like rotor thrust** at adequate resolution, with absolute drag and
torque being harder.

## The plan to validate against real RC data

These datasets are public, free, and matched to the RC regime:

- **UIUC Propeller Database** — wind-tunnel thrust/power coefficients for hundreds
  of small RC/UAV propellers across advance ratios and RPM. The reference for
  drone/RC prop validation (needs the 3D solver).
- **UIUC Low-Reynolds-Number Airfoil Database** — wind-tunnel lift/drag polars for
  low-Re airfoils. The reference for fixed-wing validation (a 2D airfoil polar is
  an earlier-reachable target).
- **NACA reports / NASA rotorcraft data** — classic public airfoil and rotor data.

When the solver can reproduce these to a few percent **at the RC Reynolds
number**, the tool will have earned the right to put a drag or thrust number in
front of a builder. Until then, treat outputs as physically correct in *behaviour*
but not yet certified in *absolute magnitude* for real-world RC conditions.
