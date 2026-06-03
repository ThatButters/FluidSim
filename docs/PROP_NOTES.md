# Propeller Mode — Design Notes & Open Items

An honest record of how propeller mode loads, orients, and measures a prop, what
the numbers mean, and **where the physics is solid vs. where it is a flagged
estimate.** Read alongside [VALIDATION.md](VALIDATION.md).

The guiding rule, as everywhere in this project: separate the quantities we trust
from the ones we don't, and say which is which.

---

## 1. Loading & orientation (any STL "just works")

The solver always spins about **+x** with the disk in the **y-z plane** (see
`prop_model.PropModel` and `_build_bank`). A propeller exported by someone else
can have its shaft on any axis; if the shaft isn't on +x, the solver spins it
about the wrong axis and the blades **tumble end-over-end** instead of sweeping a
disk.

`stl_import.orient_prop_to_spin_axis()` fixes this automatically on load:

- **Shaft detection** — `principal_frame()` takes the **area-weighted** principal
  axes of the mesh (area-weighted so it's independent of how finely each face is
  tessellated). The thinnest axis is the disk normal = the shaft.
- **Canonicalise** — the geometry is rotated so the hub sits at the origin and
  the shaft lands on +x (disk in y-z). Everything downstream (mask bank, spin,
  metrics) then works unchanged.
- **Flatness guard** — only a clearly disk-like mesh is touched; a chunky/non-prop
  mesh (flatness ratio above 0.6) is returned untouched.

### Direction & flip

- **Spin direction** — `spin="ccw"|"cw"` (or a signed number) → `spin_sign` (±1)
  applied to `omega_spin`, driving both the visual blade sweep and the moving-wall
  swirl. Live toggle: `set_spin_direction()`. Detection of a prop's *intended*
  rotation from camber is **not** attempted — it's a user choice (the CW/CCW
  toggle / the filename usually tells you).
- **Flip over** — PCA shaft direction is **sign-ambiguous**, so a prop can come in
  facing backwards. `flip=True` (or `set_flip()` / `flip_over()`) turns it 180°
  about an in-plane axis. It is a **proper rotation** (det +1), so it never
  changes handedness (CW/CCW is preserved).

### Regression

`test_orient.py` (CPU-only, no GPU/solver needed) locks all of this in: shaft
detection from arbitrary poses, canonical recovery, flip invariants (proper
rotation, no mirror), the non-prop guard, and the direction-sign mapping. It
generates test props by rotating the built-in `make_prop`, so it has **no
external-file dependency** (the real download is an optional bonus check).

---

## 2. What prop mode measures

Prop mode is an **operating-point + performance** tool, not just a flow
visualiser. The operating point is set by two controls:

- **RPM** (`u_tip`, the tip speed; capped to keep tip Mach modest for the
  incompressible solver), and
- **Flight speed** (`u_free`, the axial advance; `0` = **static / hover**).

`metrics()` reports, at that operating point:

| Quantity | Trust | Notes |
| --- | --- | --- |
| Shaft power, torque | **robust** | wall-carried angular momentum |
| C_P (power coefficient) | **robust** | built on torque/power |
| Advance ratio J = πV/u_tip | exact | kinematic |
| Wake swirl, tip Mach | robust | |
| Thrust | **estimated** | actuator-disk momentum balance; low-Re voxel caveat |
| C_T, efficiency η (cruise) | estimated | signed; η counts only propelling thrust, so windmilling reads 0 |
| Figure of merit, lift-per-power (hover) | estimated | the hover-relevant "efficiency of lift" |

**Cruise vs hover efficiency.** Advance-ratio efficiency η is **0 at hover by
definition** — useless for a lifting/drone prop. So hover reports **Figure of
Merit** (FM = ideal momentum-theory power / actual shaft power) and
**lift-per-power** instead. The UI shows whichever is relevant to the operating
point and flags the estimated ones.

**Sweeps** (`sweep_rpm` = hover lift map, `sweep_j` = cruise map; GUI button +
`demo_prop_sweep.py`) hold one variable and step the other, producing the classic
performance curves.

---

## 3. Static stability — the boundary-condition story

This is the part with the most history, and the most important caveat.

### The problem

At **static** (`u_free=0`) the original outlet — a crude zeroth-order open
outflow — let the pumped slipstream **pile up with nowhere to drain**
("closed-tank"), and the incompressible LBM **diverged** above low tip speed.
`validate_prop` is stable at high RPM *only with a real freestream* (J≈2.4), i.e.
cruise — the opposite of hover.

### What we tried (and what each taught us)

1. **Absorbing sponge** toward zero velocity near the outlet → **stable but
   choked**: it removed the slipstream momentum, so measured thrust/FM collapsed
   to ~3 %.
2. **Density-anchoring sponge** (relax ρ→1, keep velocity) → realistic at low RPM
   (FM≈27 %) but **ran away** at higher RPM (no momentum sink).
3. **Decoupled sponge** (density rescale + small velocity bleed) → still choked;
   the momentum bleed needed for stability inevitably suppresses the wake.
4. **Force-based thrust** (`compute_force`, momentum exchange on the blades) →
   **contaminated by the spinning swept-mask bookkeeping** (gave FM≈4800 %,
   scaling with tip speed, not ω²). Dead end for a swept prop.
5. **Anti-bounce-back (ABB) pressure outlet** → **fixes stability** cleanly. This
   is what shipped.

### What shipped

`LBM3D_CUDA.set_open_ends(open_inlet=False, abb_outlet=True)` enables a
**non-reflecting anti-bounce-back pressure outlet (ρ=1)**, **auto-gated in the
kernel to static** (`u_in < 1e-6`). With a real freestream the outlet velocity is
too high for the low-Mach ABB, so it falls back to the validated zeroth-order
outflow. Net:

- **Static hover runs stably at all RPM** — divergence fixed.
- **Cruise / forward flight unchanged** (gated fallback) — no regression.
- Validated wind-tunnel paths untouched (flags default **off**;
  `FlowModel`/validations never call `set_open_ends`).

### The open item — absolute hover FM is *not* physical yet

Convergence instrumentation (now removed) was decisive: the realistic-looking
**FM≈30 % was a transient**; the true steady state with this BC is **~3–5 %**.

> **Root cause:** the **velocity inlet (u=0) starves the disk.** A hovering prop
> must *draw air in* through the front, but a fixed-velocity inlet forbids it. The
> **entrainment (open) inlet** that would fix this makes the two-pressure-boundary
> system **ill-posed** — with no velocity reference anywhere, the mean flow drifts
> and diverges (confirmed: ABB at both ends blows up immediately).

So the outlet BC was **necessary but not sufficient**. Accurate absolute hover FM
is bottlenecked one layer deeper, at **inlet entrainment / domain size**, not the
outflow.

### Next step (not yet done)

A larger domain with the prop **far from the boundaries**, plus **far-field
(characteristic / LODI) boundaries on all faces** so fluid can be entrained
without an ill-posed pressure-pressure pair. This is a real piece of work, and at
this coarse, low-Re voxel fidelity the absolute-accuracy payoff is uncertain — so
it's tracked here rather than half-built.

---

## 4. How to read prop numbers today

- **Trust for absolute-ish values:** shaft power, torque, C_P, advance ratio,
  swirl, tip Mach, and all the *qualitative trends* (thrust ∝ ω² in hover; the
  signed thrust zero-crossing into windmilling as J rises).
- **Use only for relative comparison (flagged "est."):** thrust, C_T, efficiency
  η, Figure of Merit, lift-per-power. The systematic offsets largely cancel when
  comparing **two props at matched conditions**; the absolute magnitudes
  (especially hover FM, currently starved low) are **not** certified.

---

## 5. Files

| File | Role |
| --- | --- |
| `stl_import.py` | `principal_frame`, `orient_prop_to_spin_axis`, `flip_over` |
| `prop_model.py` | `PropModel`: load/auto-orient, spin direction/flip, `metrics`, `sweep_rpm`/`sweep_j` |
| `lbm3d_cuda.py` | `set_open_ends` (ABB pressure outlet / entrainment inlet, gated); legacy `set_sponge` retained but unused |
| `fluidsim_gui.py` | prop controls: direction toggle, flip checkbox, performance readout, sweep button |
| `demo_prop_sweep.py` | headless `--mode hover|cruise` performance sweep + plot |
| `test_orient.py` | CPU regression for orientation/flip/direction |

> Note: `set_sponge`/`_apply_sponge` in `lbm3d_cuda.py` are **superseded** by the
> ABB outlet and are no longer called. Kept (opt-in, off by default) in case an
> absorbing layer is wanted for a future far-field setup; safe to delete if not.
