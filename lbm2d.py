"""
lbm2d -- A 2D Lattice Boltzmann (D2Q9, BGK) reference solver.

This is the *reference implementation* for the FluidSim project. Its job is
correctness, establishing ground-truth physics (validated against published
wind-tunnel / DNS values).

It is **backend-agnostic**: pass ``array_module=cupy`` to run the identical code
on an NVIDIA GPU, or leave the NumPy default for the CPU reference. The two must
produce the same numbers (see gpu_benchmark.py), so the CPU version stays the
golden oracle for the GPU port.

D2Q9 lattice, velocity ordering (index : velocity), chosen so that the
opposite of direction ``i`` is always ``8 - i`` (handy for bounce-back):

    0:( 1, 1)  1:( 1, 0)  2:( 1,-1)
    3:( 0, 1)  4:( 0, 0)  5:( 0,-1)
    6:(-1, 1)  7:(-1, 0)  8:(-1,-1)

Core algorithm (Bhatnagar-Gross-Krook single-relaxation-time):
    macroscopic -> equilibrium -> collide -> boundary -> stream

The structure deliberately mirrors Jonas Latt's canonical Palabos cylinder
example (a known-good baseline), repackaged as a reusable class with on-body
force extraction added.
"""

from __future__ import annotations

import numpy as np

# --- D2Q9 lattice constants -------------------------------------------------
# Velocity vectors c_i (column = [cx, cy]); ordering gives opposite(i) == 8 - i.
C = np.array(
    [[1, 1], [1, 0], [1, -1],
     [0, 1], [0, 0], [0, -1],
     [-1, 1], [-1, 0], [-1, -1]],
    dtype=np.int64,
)
# Lattice weights w_i (sum to 1).
W = np.array([1 / 36, 1 / 9, 1 / 36,
              1 / 9, 4 / 9, 1 / 9,
              1 / 36, 1 / 9, 1 / 36])
OPP = np.array([8 - i for i in range(9)])           # opposite-direction index
# Direction-index groups by sign of cx, used for the inlet/outlet BCs. Plain
# Python lists so they index both NumPy and CuPy arrays without conversion.
COL_PLUS = [0, 1, 2]      # cx = +1  (rightward)
COL_ZERO = [3, 4, 5]      # cx =  0
COL_MINUS = [6, 7, 8]     # cx = -1  (leftward)
COL_MINUS_REV = [8, 7, 6]  # COL_MINUS reversed (== opposite of COL_PLUS)


class LBM2D:
    """A 2D single-relaxation-time (BGK) lattice-Boltzmann fluid.

    Parameters
    ----------
    nx, ny : int
        Domain size in lattice units.
    u_lb : float
        Inflow speed in lattice units (keep << 1 to stay low-Mach; ~0.05 good).
    re : float
        Reynolds number based on the characteristic length ``char_length``.
    char_length : float
        Characteristic length (e.g. cylinder diameter) in lattice units, used
        to set viscosity from ``re`` and to non-dimensionalise forces.
    """

    def __init__(self, nx: int, ny: int, *, u_lb: float, re: float,
                 char_length: float, array_module=np,
                 collision: str = "bgk") -> None:
        self.nx, self.ny = nx, ny
        self.u_lb = u_lb
        self.re = re
        self.char_length = char_length
        self.xp = array_module               # numpy (CPU) or cupy (GPU)
        self.collision = collision           # "bgk" or "regularized"
        xp = self.xp

        # Viscosity from Re (Re = U * L / nu), then BGK relaxation rate.
        self.nu = u_lb * char_length / re
        self.omega = 1.0 / (3.0 * self.nu + 0.5)   # must be < 2 for stability

        # solid[x, y] = True marks obstacle (no-slip bounce-back) cells.
        self.solid = xp.zeros((nx, ny), dtype=bool)
        # body, if set, restricts force measurement to a subset of solid cells
        # (e.g. the cylinder only, excluding channel walls). None -> use solid.
        self.body = None
        # rot_center, if set, enables torque measurement about that point.
        self.rot_center = None
        self._boundary_torque = 0.0
        # Compute surface force/torque every step. Forces involve host-side
        # checks that stall the GPU pipeline; set False to skip them on steps
        # where the load isn't needed (sample periodically instead).
        self.measure_force = True
        # cell-coordinate grids (for torque moment arms).
        self._X, self._Y = xp.meshgrid(xp.arange(nx), xp.arange(ny),
                                       indexing="ij")

        # Steady inflow profile (2, nx, ny), enforced at the inlet each step.
        # A small antisymmetric sinusoidal vx(y) perturbation breaks up/down
        # symmetry so the Karman instability grows and sheds in finite time
        # (saturated Cd/St are independent of this seed amplitude).
        self.vel = xp.zeros((2, nx, ny))
        self.vel[0] = u_lb * (1.0 + 5e-3 * xp.sin(
            xp.linspace(0, 2 * np.pi, ny)[None, :].repeat(nx, axis=0)))

        # Distribution functions f_i(x, y); start at equilibrium of the inflow.
        self.f = self._equilibrium(xp.ones((nx, ny)), self.vel)
        self._boundary_force = xp.zeros(2)    # last computed [Fx, Fy] (lattice)

    # -- obstacle helpers ----------------------------------------------------
    def add_circle(self, cx: float, cy: float, r: float) -> None:
        """Mark a filled circle of radius ``r`` at (cx, cy) as solid."""
        x = self.xp.arange(self.nx)[:, None]
        y = self.xp.arange(self.ny)[None, :]
        self.solid |= (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

    # -- core LBM steps ------------------------------------------------------
    def _equilibrium(self, rho, u):
        """Second-order Maxwell-Boltzmann equilibrium f_i^eq(rho, u)."""
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
        feq = self.xp.empty((9, self.nx, self.ny))
        for i in range(9):
            cu = 3.0 * (C[i, 0] * u[0] + C[i, 1] * u[1])
            feq[i] = rho * W[i] * (1.0 + cu + 0.5 * cu ** 2 - usqr)
        return feq

    def macroscopic(self):
        """Return (rho, u) reduced from the current distributions."""
        rho = self.f.sum(axis=0)
        u = self.xp.zeros((2, self.nx, self.ny))
        for i in range(9):
            u[0] += C[i, 0] * self.f[i]
            u[1] += C[i, 1] * self.f[i]
        u /= rho
        return rho, u

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        # 1. Outlet (right wall): zeroth-order outflow -- copy the leftward-
        #    moving populations from the next column inward.
        self.f[COL_MINUS, -1, :] = self.f[COL_MINUS, -2, :]

        # 2. Macroscopic fields.
        rho, u = self.macroscopic()

        # 3. Inlet (left wall): Zou/He velocity boundary condition.
        u[:, 0, :] = self.vel[:, 0, :]
        rho[0, :] = (1.0 / (1.0 - u[0, 0, :])) * (
            self.f[COL_ZERO, 0, :].sum(axis=0)
            + 2.0 * self.f[COL_MINUS, 0, :].sum(axis=0))

        # 4. Equilibrium and the unknown inbound inlet populations.
        feq = self._equilibrium(rho, u)
        self.f[COL_PLUS, 0, :] = (feq[COL_PLUS, 0, :]
                                  + self.f[COL_MINUS_REV, 0, :]
                                  - feq[COL_MINUS_REV, 0, :])

        # 5. Collision: BGK, regularised, or regularised + Smagorinsky LES.
        if self.collision == "les":
            fout = self._collide_regularized(feq, omega=self._les_omega(feq))
        elif self.collision == "regularized":
            fout = self._collide_regularized(feq)
        else:
            fout = self.f - self.omega * (self.f - feq)

        # 6. No-slip bounce-back on the obstacle (and measure the force on it
        #    BEFORE overwriting, via the momentum-exchange method).
        if self.measure_force:
            self._fout = fout                # cached (post-collision) for diag
            self._boundary_force = self._momentum_exchange_force(fout)
        self._apply_bounceback(fout)

        # 7. Streaming: shift each population along its velocity vector.
        for i in range(9):
            self.f[i] = self.xp.roll(self.xp.roll(fout[i], C[i, 0], axis=0),
                                     C[i, 1], axis=1)

    def _apply_bounceback(self, fout: np.ndarray) -> None:
        """No-slip (stationary-wall) bounce-back at solid nodes.

        Overridden by moving-boundary subclasses to add a wall-velocity term.
        """
        for i in range(9):
            fout[i, self.solid] = self.f[OPP[i], self.solid]

    def _momentum_flux(self, feq):
        """Non-equilibrium momentum-flux tensor components (Pxx, Pxy, Pyy)."""
        xp = self.xp
        fneq = self.f - feq
        pxx = xp.zeros((self.nx, self.ny))
        pxy = xp.zeros((self.nx, self.ny))
        pyy = xp.zeros((self.nx, self.ny))
        for i in range(9):
            if i == 4:
                continue
            pxx += (C[i, 0] * C[i, 0]) * fneq[i]
            pxy += (C[i, 0] * C[i, 1]) * fneq[i]
            pyy += (C[i, 1] * C[i, 1]) * fneq[i]
        return pxx, pxy, pyy

    def _collide_regularized(self, feq, omega=None):
        """Regularised collision: rebuild the non-equilibrium populations from
        their momentum-flux tensor before relaxing, filtering the higher-order
        "ghost" modes that destabilise BGK as the viscosity -> 0 (high Re).

        f_neq_reg_i = (9/2) w_i [ (c_i . Pi . c_i) - (1/3) tr(Pi) ],   Pi the
        non-equilibrium momentum-flux tensor; then f_i = f_eq_i + (1-omega) f_neq_reg_i.
        ``omega`` may be a per-cell field (Smagorinsky LES).
        """
        if omega is None:
            omega = self.omega
        pxx, pxy, pyy = self._momentum_flux(feq)
        tr3 = (pxx + pyy) / 3.0
        one_minus = 1.0 - omega
        fout = self.xp.empty_like(self.f)
        for i in range(9):
            cpc = (C[i, 0] * C[i, 0] * pxx
                   + 2.0 * C[i, 0] * C[i, 1] * pxy
                   + C[i, 1] * C[i, 1] * pyy)
            fneq_reg = 4.5 * W[i] * (cpc - tr3)
            fout[i] = feq[i] + one_minus * fneq_reg
        return fout

    def _les_omega(self, feq, c_smag: float = 0.16):
        """Per-cell relaxation rate with a Smagorinsky eddy viscosity.

        The local strain follows from the non-equilibrium stress |Pi|, so the
        total relaxation time solves
            tau_t = 0.5 (tau0 + sqrt(tau0^2 + 18*sqrt2 * Cs^2 * |Pi| / rho)),
        adding dissipation where the flow is straining hardest -- which both
        models unresolved turbulence (LES) and stabilises high-Re flow.
        """
        xp = self.xp
        pxx, pxy, pyy = self._momentum_flux(feq)
        pi_mag = xp.sqrt(pxx ** 2 + pyy ** 2 + 2.0 * pxy ** 2)
        rho = self.f.sum(axis=0)
        tau0 = 1.0 / self.omega
        tau_t = 0.5 * (tau0 + xp.sqrt(
            tau0 ** 2 + 18.0 * np.sqrt(2.0) * c_smag ** 2 * pi_mag / rho))
        return 1.0 / tau_t

    # -- force extraction (the heart of every downstream metric) -------------
    def _momentum_exchange_force(self, fout: np.ndarray) -> np.ndarray:
        """Total fluid->solid force [Fx, Fy] in lattice units.

        Momentum-exchange method for halfway bounce-back. For each interface
        link (fluid node x_f whose neighbour x_f + c_i is solid), the population
        f_i carries momentum c_i*f_i into the wall and is reflected as the
        bounced population (= f_i), so the momentum delivered to the body is
        2*c_i*f_i per link. Summing over all links gives the net force -- the
        discrete analogue of integrating pressure + shear over the surface, and
        the basis for all downstream lift / drag / thrust.

        (An earlier variant paired f_i with the fluid node's OWN opposite
        population f_opp instead of the bounced population; that under-predicts
        Cd by ~20% -- see diag_force.py for the benchmark that settled this.)
        """
        body = self.solid if self.body is None else self.body
        force = self.xp.zeros(2)
        torque = 0.0
        do_torque = self.rot_center is not None
        if do_torque:
            cx, cy = self.rot_center
        for i in range(9):
            if i == 4:                       # rest population carries no link
                continue
            # Fluid nodes whose i-neighbour is body == interface links. The
            # neighbour test uses `body` (what we measure) while the fluid test
            # uses `solid` (everything no-slip), so channel walls bounce the
            # flow but don't contribute to the measured force.
            neigh_body = self.xp.roll(self.xp.roll(body, -C[i, 0], axis=0),
                                      -C[i, 1], axis=1)
            link = neigh_body & ~self.solid
            if not bool(link.any()):
                continue
            # Momentum delivered per link (2 c_i f_i). This is the validated
            # stationary-wall form (see the DFG benchmark). NOTE: extracting an
            # accurate *net* load on a fast-moving, re-voxelising boundary is a
            # known-hard problem -- the raw exchange is dominated by the momentum
            # the wall convects, not the aerodynamic drag -- so for moving bodies
            # we validate via the angular-momentum budget instead (see
            # demo_rotor.py), and treat moving-boundary load extraction as a
            # later calibration task.
            momentum = 2.0 * fout[i][link]
            dfx, dfy = C[i, 0] * momentum, C[i, 1] * momentum
            force[0] += dfx.sum()
            force[1] += dfy.sum()
            if do_torque:                                # tau_z = r x F
                rx = self._X[link] - cx
                ry = self._Y[link] - cy
                torque += float((rx * dfy - ry * dfx).sum())
        self._boundary_torque = torque
        return force

    @property
    def force(self) -> np.ndarray:
        """Last computed [Fx, Fy] on the obstacle (lattice units)."""
        return self._boundary_force

    @property
    def torque(self) -> float:
        """Last computed torque on the body about rot_center (lattice units)."""
        return self._boundary_torque

    def to_host(self, arr):
        """Return a NumPy copy of an array (no-op on the CPU backend)."""
        return arr if self.xp is np else self.xp.asnumpy(arr)

    def coefficients(self) -> tuple[float, float]:
        """(Cd, Cl) from the last force, normalised by 0.5*rho*U^2*L (rho=1)."""
        denom = 0.5 * self.u_lb ** 2 * self.char_length
        return (float(self._boundary_force[0]) / denom,
                float(self._boundary_force[1]) / denom)

    def vorticity(self) -> np.ndarray:
        """Scalar out-of-plane vorticity dvy/dx - dvx/dy (host array, for plots)."""
        _, u = self.macroscopic()
        dvy_dx = self.xp.gradient(u[1], axis=0)
        dvx_dy = self.xp.gradient(u[0], axis=1)
        w = dvy_dx - dvx_dy
        w[self.solid] = float("nan")    # blank the body in plots
        return self.to_host(w)
