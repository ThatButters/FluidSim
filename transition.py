"""
transition -- a transported-intermittency transition model for the LBM solver.

The fix for what the local model lacked: MEMORY. The intermittency gamma is a
scalar field advected and diffused WITH the flow, so once the boundary layer
trips turbulent it stays turbulent downstream and reattaches the separation
bubble -- exactly the history effect a local switch cannot capture.

gamma evolves by advection-diffusion-reaction (a one-equation Menter-style model):

    d(gamma)/dt + u.grad(gamma) = D lap(gamma) + P,     P = c_prod S F_onset (1-gamma)

Transition onset F_onset fires where the vorticity Reynolds number
Re_v = d^2 S / nu exceeds a critical value (d = wall distance). Because Re_v
carries d^2, it is small in the thin attached boundary layer (small d) and large
in the detached shear layer (large d) -- so transition triggers in the separated
shear layer, not the attached run, which is the whole point.

Laminar inflow (gamma=0) and gamma=0 in solids are enforced each substep. The
field is consumed by the solver to gate its LES eddy viscosity.
"""

from __future__ import annotations

import cupy as cp


def strain_magnitude(u):
    ux, uy = u[0], u[1]
    sxx = cp.gradient(ux, axis=0)
    syy = cp.gradient(uy, axis=1)
    sxy = 0.5 * (cp.gradient(ux, axis=1) + cp.gradient(uy, axis=0))
    return cp.sqrt(2.0 * (sxx ** 2 + syy ** 2 + 2.0 * sxy ** 2)) + 1e-12


class GammaTransport:
    def __init__(self, dist, nu_mol, *, rev_crit=200.0, c_prod=0.04,
                 diff=0.06, gamma0=0.0):
        self.dist = dist.astype(cp.float32)
        self.nu_mol = float(nu_mol)
        self.rev_crit = float(rev_crit)
        self.c_prod = float(c_prod)
        self.diff = float(diff)
        self.gamma = cp.full(dist.shape, gamma0, dtype=cp.float32)

    def f_onset(self, S):
        rev = self.dist ** 2 * S / self.nu_mol
        return cp.clip((rev - self.rev_crit) / (0.3 * self.rev_crit), 0.0, 1.0)

    def update(self, u, solid, substeps=1):
        ux, uy = u[0], u[1]
        S = strain_magnitude(u)
        F = self.f_onset(S)
        prod_base = self.c_prod * S * F
        g = self.gamma
        for _ in range(substeps):
            # first-order upwind advection u . grad(gamma)
            gxm = g - cp.roll(g, 1, 0)
            gxp = cp.roll(g, -1, 0) - g
            gym = g - cp.roll(g, 1, 1)
            gyp = cp.roll(g, -1, 1) - g
            adv = cp.where(ux > 0, ux * gxm, ux * gxp) \
                + cp.where(uy > 0, uy * gym, uy * gyp)
            lap = (cp.roll(g, 1, 0) + cp.roll(g, -1, 0)
                   + cp.roll(g, 1, 1) + cp.roll(g, -1, 1) - 4.0 * g)
            g = g + (-adv + self.diff * lap + prod_base * (1.0 - g))
            g = cp.clip(g, 0.0, 1.0)
            g[0, :] = 0.0                 # laminar inflow
            g[solid] = 0.0               # no intermittency inside the body
        self.gamma = g.astype(cp.float32)
        return self.gamma


class TransitionSim:
    """Couples an LBM2D_CUDA solver (collision='transition') to GammaTransport:
    each step runs the flow; every `gamma_every` steps the intermittency is
    advanced from the current velocity field and fed back to gate the LES."""

    def __init__(self, sim, *, rev_crit=200.0, c_prod=0.04, diff=0.06,
                 gamma_every=3):
        from cupyx.scipy import ndimage as cnd
        dist = cnd.distance_transform_edt(sim.solid == 0).astype(cp.float32)
        tau0 = 1.0 / float(sim.omega)
        nu_mol = (tau0 - 0.5) / 3.0
        self.sim = sim
        self.gt = GammaTransport(dist, nu_mol, rev_crit=rev_crit,
                                 c_prod=c_prod, diff=diff)
        self.solid_bool = sim.solid.astype(bool)
        self.gamma_every = gamma_every
        self.t = 0
        sim.gamma = self.gt.gamma

    def step(self):
        self.sim.step()
        self.t += 1
        if self.t % self.gamma_every == 0:
            _, u = self.sim.macroscopic()
            self.sim.gamma = self.gt.update(u, self.solid_bool,
                                            substeps=self.gamma_every)
