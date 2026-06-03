# FluidSim — Technical Overview

This document explains **what we're building**, **why**, **the technology stack**,
and **the math** behind the solver we are writing from scratch. It assumes
curiosity more than a CFD background — the goal is that an RC hobbyist *and* an
engineer both come away understanding how the thing works.

---

## 1. The goal

FluidSim is a **free, open-source GPU "wind tunnel"** for radio-control aircraft.
You import an `.stl` of your design — a wing, a fuselage, a propeller, a
quadcopter frame, a helicopter rotor — and your NVIDIA GPU simulates the air
flowing around it. You see the **flow itself** (streamlines, wakes, vortices,
surface pressure) and you get the **engineering numbers** that decide whether a
design is good: lift, drag, thrust, and efficiency.

It targets three communities equally — **RC airplanes, RC helicopters, and
multirotor drones** — because they share one physics engine but ask different
questions of it.

**Why build it at all?** Because nothing free does this combination today:

| Existing option | What it can't do |
| --- | --- |
| Cloud CFD (SimScale, AirShaper) | Paid, not real-time, runs on remote servers |
| Hobbyist tools (XFLR5, OpenVSP) | Can't ingest an arbitrary STL; can't resolve a spinning prop's wake |
| FluidX3D (the one capable free GPU engine) | Non-commercial license; no hobbyist UI or analytics |

The unfilled niche is precisely: **drop-in STL + real-time GPU flow +
hobbyist-friendly + planes *and* helis *and* drones + force analytics**, under a
permissive license. That's what we're filling.

---

## 2. The technology stack

We're deliberately building in **two stages**: a correctness-first reference
implementation, then a performance-first production engine. They serve different
masters — one must be *obviously right*, the other *fast*.

### Stage 1 — Reference solver (what exists today)

| Layer | Choice | Why |
| --- | --- | --- |
| Language | **Python 3** | Fast to write, easy to read, easy to verify |
| Compute | **NumPy** (vectorised CPU arrays) | No GPU dependency; the goal here is *correct physics*, not speed |
| Visualisation | **Matplotlib** | Quick vorticity frames and validation plots |

The reference solver is the project's **ground truth**. It is slow (tens of
timesteps per second on a CPU), and that's fine: its entire job is to reproduce
textbook results exactly, so that every future optimisation can be checked
against a known-correct oracle. It is also **framework-agnostic** — no CUDA, no
exotic dependencies — so the physics is decoupled from the hardware.

> A practical note: the target machine runs **Python 3.14**, which is too new for
> GPU-Python frameworks like Taichi (no wheels yet). That pushed us toward a pure
> NumPy reference now and a **native** GPU engine later — which is the right call
> for performance anyway.

### Stage 2 — Production engine (planned)

| Layer | Planned choice | Why |
| --- | --- | --- |
| Solver | **Native CUDA / C++** | LBM is memory-bandwidth-bound; native code with a packed memory layout (à la FluidX3D: ~55 bytes/cell) targets ~10,000 MLUPS and ~300M cells in 16 GB on an RTX 5070 Ti |
| Collision model | **MRT / cumulant** (upgrade from BGK) | Plain BGK destabilises at high Reynolds number; better collision operators stay stable on coarser grids |
| Geometry | **GPU STL voxelisation** | Convert an arbitrary triangle mesh to lattice cells directly on the GPU |
| Rendering | **OpenGL / Vulkan interop** | Share GPU buffers with the solver (zero-copy) for smooth real-time flow visualisation |

This separation means the hard, error-prone physics gets nailed down once, in a
form anyone can read, before we chase the 100×+ speedups that make it
interactive.

---

## 3. The math — what we're building from scratch

We do **not** solve the Navier–Stokes equations directly. Instead we use the
**Lattice Boltzmann Method (LBM)**, which arrives at the same fluid behaviour
from a different, much more GPU-friendly direction.

### 3.1 The idea: track particle *populations*, not the fluid directly

Classical CFD treats the fluid as a continuum and solves for velocity and
pressure fields directly — which requires solving a global pressure equation
that couples every cell to every other cell (awkward on a GPU).

LBM instead borrows from **kinetic theory**. Imagine the fluid as a huge number
of particles streaming and colliding. We don't track individual particles — we
track a **distribution function** $f(\mathbf{x}, \boldsymbol{\xi}, t)$: roughly,
"how many particles are near position $\mathbf{x}$ moving with velocity
$\boldsymbol{\xi}$ at time $t$." Its evolution is the **Boltzmann transport
equation**:

$$\frac{\partial f}{\partial t} + \boldsymbol{\xi}\cdot\nabla f = \Omega(f)$$

The left side is **streaming** (particles drift along their velocity); the right
side $\Omega$ is **collisions** (particles redistribute). The magic is that
fluid behaviour — pressure, viscosity, vortices — *emerges* from this simple
streaming-and-colliding picture.

### 3.2 Discretising velocity: the D2Q9 lattice

A computer can't track every possible velocity $\boldsymbol{\xi}$. The key LBM
insight is that we only need a **small, fixed set** of velocities. In 2D we use
**D2Q9** — *9 velocities* per lattice node: one rest particle, four pointing
along the axes, four along the diagonals.

```
   6   3   0          c6 c3 c0      diagonals : c0,c2,c6,c8   (speed √2)
     ↖ ↑ ↗            axial     : c1,c3,c5,c7   (speed 1)
   7 ← 4 → 1          rest      : c4            (speed 0)
     ↙ ↓ ↘
   8   5   2
```

Each velocity $\mathbf{c}_i$ has a **weight** $w_i$ (how much of the equilibrium
it carries):

$$w_4 = \tfrac{4}{9}\ (\text{rest}),\quad w_{1,3,5,7} = \tfrac{1}{9}\ (\text{axial}),\quad w_{0,2,6,8} = \tfrac{1}{36}\ (\text{diagonal})$$

So at every grid cell we store **9 numbers** $f_i$ — the particle populations
heading in each of the 9 directions. That's the entire state of the fluid.

### 3.3 The algorithm: collide, then stream

Discretising the Boltzmann equation in space and time gives the **lattice
Boltzmann equation**:

$$f_i(\mathbf{x} + \mathbf{c}_i\,\Delta t,\ t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau}\Big[f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\mathbf{x}, t)\Big]$$

which we apply as two cheap, **local** steps every timestep:

**1. Collision** (relax each population toward local equilibrium):

$$f_i^{*}(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \omega\,\big[f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\mathbf{x}, t)\big], \qquad \omega = \frac{1}{\tau}$$

**2. Streaming** (shove each population to its neighbour in direction
$\mathbf{c}_i$):

$$f_i(\mathbf{x} + \mathbf{c}_i,\ t+1) = f_i^{*}(\mathbf{x}, t)$$

That's it. Both steps touch only a cell and its immediate neighbours — **no
global solve, no all-to-all coupling.** This locality is *exactly* why LBM maps
so well onto thousands of GPU cores: every cell can be updated independently, in
parallel.

### 3.4 The equilibrium distribution

The collision step relaxes toward the **discrete Maxwell–Boltzmann equilibrium**,
expanded to second order in the velocity (valid at low Mach number):

$$f_i^{\text{eq}} = w_i\,\rho\left[\,1 + \frac{\mathbf{c}_i\cdot\mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i\cdot\mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u}\cdot\mathbf{u}}{2c_s^2}\,\right]$$

For the D2Q9 lattice the **speed of sound** is $c_s^2 = \tfrac{1}{3}$, so in code
this becomes the familiar:

$$f_i^{\text{eq}} = w_i\,\rho\left[\,1 + 3(\mathbf{c}_i\cdot\mathbf{u}) + \tfrac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2 - \tfrac{3}{2}\,\mathbf{u}\cdot\mathbf{u}\,\right]$$

### 3.5 Getting real physics back out

The macroscopic quantities you actually care about are just **moments** (sums)
of the 9 populations:

$$\rho = \sum_i f_i \qquad\text{(density / pressure, since } p = c_s^2\rho)$$

$$\rho\,\mathbf{u} = \sum_i \mathbf{c}_i\,f_i \qquad\text{(momentum} \Rightarrow \text{velocity)}$$

And here is the remarkable part, provable by a **Chapman–Enskog expansion**: in
the limit of low Mach number, this streaming-and-colliding of fictitious
particles recovers the **incompressible Navier–Stokes equations** — the real
equations of fluid motion. LBM isn't an approximation *of* a fluid solver; it's
a different, equivalent route to the same physics.

### 3.6 Viscosity, Reynolds number, and the relaxation time

The single relaxation parameter $\tau$ (or $\omega = 1/\tau$) sets the fluid's
**kinematic viscosity**:

$$\nu = c_s^2\left(\tau - \tfrac{1}{2}\right) = \frac{1}{3}\left(\frac{1}{\omega} - \frac{1}{2}\right)$$

and the **Reynolds number** — the ratio of inertial to viscous forces, which
governs whether flow is smooth or turbulent — is

$$\mathrm{Re} = \frac{U L}{\nu}$$

with $U$ a characteristic speed and $L$ a characteristic length (e.g. chord or
diameter). RC aircraft live at $\mathrm{Re}\sim 10^4$–$10^5$ — the tricky regime
where flow transitions from laminar to turbulent, which is a major reason this
problem is interesting and hard.

### 3.7 The incompressibility (Mach) limit

LBM is only valid when the flow speed is small compared with the lattice speed
of sound. The **Mach number** is

$$\mathrm{Ma} = \frac{U}{c_s} = U\sqrt{3}$$

and compressibility error grows like $\mathcal{O}(\mathrm{Ma}^2)$. In practice we
keep lattice velocities $U \lesssim 0.05$ (so $\mathrm{Ma} \lesssim 0.09$). We
learned this concretely: a test at $U=0.08$ pushed the *local* Mach past ~0.15
around an obstacle and visibly corrupted the forces. Staying slow keeps the
physics honest.

### 3.8 Boundaries: how the fluid feels a solid

Geometry enters through **boundary conditions** at solid cells.

**No-slip walls — bounce-back.** A population that would stream into a solid is
simply reflected back the way it came:

$$f_{\bar{\imath}}(\mathbf{x}_f,\ t+1) = f_i^{*}(\mathbf{x}_f,\ t)$$

where $\bar{\imath}$ is the direction opposite $i$. Done at the midpoint between
fluid and solid ("halfway" bounce-back), this is second-order accurate and
handles **arbitrary voxelised shapes** — which is exactly why an imported STL
"just works": voxelise it, mark those cells solid, bounce-back.

**Moving / rotating walls.** For a surface moving with velocity $\mathbf{u}_w$
(a spinning propeller!), the reflected population gets a **momentum kick**:

$$f_{\bar{\imath}} = f_i^{*} + \frac{2\,w_i\,\rho_w\,(\mathbf{c}_i\cdot\mathbf{u}_w)}{c_s^2} = f_i^{*} + 6\,w_i\,\rho_w\,(\mathbf{c}_i\cdot\mathbf{u}_w)$$

This single extra term is what lets us simulate rotors and props — the
make-or-break capability of the whole project. We validated it against
**Taylor–Couette flow** (a rotating cylinder with an exact analytical answer) to
**0.29 % RMS error**.

### 3.9 Forces: where lift, drag and thrust come from

Every number an RC designer wants is a **force on the body**, and LBM gives it
almost for free via the **momentum-exchange method**. At each boundary link
(a fluid cell whose neighbour is solid), the population $f_i$ carries momentum
$\mathbf{c}_i f_i$ into the wall and bounces back carrying the opposite — so the
momentum delivered to the body per link is $2\,\mathbf{c}_i f_i^{*}$. Summing
over the whole surface:

$$\mathbf{F} = \sum_{\text{boundary links}} 2\,\mathbf{c}_i\,f_i^{*}(\mathbf{x}_f)$$

This is the discrete equivalent of integrating pressure and shear stress over the
surface. From $\mathbf{F}$ we get the standard coefficients:

$$C_d = \frac{2 F_x}{\rho\,U^2\,L}, \qquad C_l = \frac{2 F_y}{\rho\,U^2\,L}$$

— drag and lift — and, for rotating cases, thrust and torque. (Getting this
formula *exactly* right mattered: an earlier variant under-predicted drag by
~20 %. We pinned down the correct form by benchmarking against the known
Re = 100 cylinder; see `diag_force.py`.)

---

## 4. What's proven, and what's next

**Validated in the reference solver:**

- Correct unsteady wake physics — the **Kármán vortex street** reproduces the
  published shedding frequency (Strouhal number), confirmed two independent ways.
- Correct **surface forces** — the momentum-exchange drag formula, benchmarked
  against the Re = 100 cylinder, and then confirmed *exactly* against the
  **Schäfer–Turek (DFG) benchmark**: $C_{d,\max}=3.28$, $C_{l,\max}=0.94$,
  $\mathrm{St}=0.306$, all matching the published reference values.
- Correct **rotating-boundary physics** — Taylor–Couette to 0.29 % RMS vs the
  exact solution. *This is the hard part, and it works.*

**The road to a usable tool** (see the README roadmap): truly sweeping rotating
geometry (a blade that physically moves through the grid), then the native-CUDA
GPU port for real-time speed, then 3D + STL import + interactive visualisation,
then the per-domain analytics that turn a flow field into the answers a plane,
heli, or drone builder actually asks.

The strategy throughout: **prove the physics in something anyone can read, then
make it fast.**
