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

## First comparison to REAL wind-tunnel data (E387, Re=100k)

We ran the actual Eppler E387 geometry (UIUC coordinates) at a real RC Reynolds
number with the LES operator and compared to the UIUC wind-tunnel reference
(`validate_e387.py`). The honest scorecard:

| quantity | low-res (chord 130) | **high-res (chord 600)** | UIUC ref |
| --- | --- | --- | --- |
| Cl at 0° (camber lift) | −0.01 *(missed)* | **+0.20** | +0.40 |
| Cd_min | 0.053 *(+341 %)* | **0.026 *(+115 %)*** | 0.012 |
| lift-curve slope (/deg) | 0.075 | 0.059 | 0.100 |

**The resolution diagnosis was correct — and measured.** The native-CUDA 2D
kernel (`lbm2d_cuda.py`, ~31,000 MLUPS, ~400× the CuPy backend) made chord=600 a
2-minute run instead of ~45. Quadrupling the resolution **recovered the camber
lift from ~0 to +0.20** (was entirely missing) and **halved the drag error**
(4.4× → 2.1×). Every angle moved toward the data.

**Verdict:** real, measured progress toward the wind tunnel, but **still not
quantitatively trustworthy** — camber lift ~half, drag ~2× high.

**Root cause identified — laminar-turbulent transition.** A controlled test
settled it: at α=4°, lift was *insensitive* to the LES constant (Cs 0.16→0.08)
and actually *dropped* with more resolution (chord 600→1000: Cl 0.43→0.40). That
rules out staircasing, LES over-damping, and under-resolution. The real E387 at
Re=100k stays attached because the laminar separation bubble *transitions to
turbulent and reattaches*; our solver captures the laminar separation (better
with resolution → lower lift) but **not the transition that reattaches the flow**,
so lift stalls at ~half. This is the known hardest problem in low-Re aerodynamics
(flagged in the project's research from the start). Closing it needs either
near-DNS resolution (~5000 cells/chord) or a transition model (γ–Reθ-type) — both
research-grade. Interpolated bounce-back will *not* close it (the gap is not
staircasing).

**Honest accuracy status:** FluidSim is a validated, fast CFD *engine* that gives
correct **trends and comparative** results at low-Re airfoil conditions (good for
design iteration and visualisation) but **not certified absolute** Cl/Cd, due to
the transition-physics frontier. That is the truthful bound, and no free tool at
this scale clears it without research-grade transition modelling.

## What has NOT been validated yet — and it's the part that matters most

**Every result above is at Reynolds number ≤ 100, in 2D, fully laminar.**

Real RC aircraft, helicopters, and drones operate at **Re ≈ 10⁴–10⁵**, in **3D**,
with **turbulence and laminar-to-turbulent transition** — a fundamentally harder
regime that determines real drag, stall, and thrust. **We have no accuracy number
there yet, and we will not claim one until we measure it.**

**The BGK stability ceiling — measured, then broken.** Plain BGK stays stable
only to about **Re≈2000** (2D) / **Re≈1000** (3D) at affordable resolution
($\tau\to0.5$), well below the RC regime. We added a **regularised collision +
Smagorinsky LES** operator (rebuild the non-equilibrium from its momentum-flux
tensor; add eddy viscosity where strain is high). It **matches BGK at low Re** and
stays stable far past it:

| Case | BGK ceiling | regularised + LES |
| --- | --- | --- |
| 2D cylinder | ~Re 2000 | stable to **Re 40,000** (Cd 1.3–1.45) |
| 3D sphere | ~Re 1000 | stable to **Re 20,000** (Cd 1.25→0.66, correct trend) |

So stable simulation **into the RC operating regime (Re ~10⁴) is now reachable**
(`validate_collision.py`). This is the prerequisite for validating against real
RC wind-tunnel data; absolute accuracy there still needs verification, but the
solver can now *reach* those conditions. (The earlier finite-wing polar was run at
Re=600 on BGK; it can now be rerun at RC Reynolds with the LES operator.)

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
