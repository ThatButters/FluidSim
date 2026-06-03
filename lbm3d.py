"""
lbm3d -- A 3D Lattice Boltzmann (D3Q19, BGK) solver.

The direct 3D extension of the validated 2D core (lbm2d.py): same collide-stream
algorithm, same equilibrium, same momentum-exchange forces -- now with 19
velocities in 3D instead of 9 in 2D. Backend-agnostic (``array_module=numpy`` for
CPU, ``cupy`` for GPU); defaults to float32, the practical choice for 3D where
memory is the binding constraint.

D3Q19 velocity set: 1 rest + 6 axial (speed 1) + 12 edge-diagonal (speed sqrt2).
Weights: rest 1/3, axial 1/18, diagonal 1/36 (sum to 1; c_s^2 = 1/3).
"""

from __future__ import annotations

import numpy as np

# --- D3Q19 lattice constants ------------------------------------------------
# Velocities paired (i, i+1) as opposites for easy bounce-back.
C3 = np.array([
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0),
    (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1),
    (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1),
    (0, 1, -1), (0, -1, 1),
], dtype=np.int64)

W3 = np.array([1 / 3]
              + [1 / 18] * 6
              + [1 / 36] * 12)

# Opposite-direction index (so streaming back is a bounce-back).
OPP3 = np.array([int(np.where((C3 == -C3[i]).all(axis=1))[0][0])
                 for i in range(19)])

# Index groups by sign of cx, for the inlet/outlet boundary conditions.
CX_POS = [i for i in range(19) if C3[i, 0] == 1]    # entering from the inlet
CX_NEG = [i for i in range(19) if C3[i, 0] == -1]   # leaving at the outlet


def _roll3(xp, a, c):
    """Shift a 3D array by lattice vector c = (cx, cy, cz)."""
    return xp.roll(xp.roll(xp.roll(a, int(c[0]), axis=0),
                           int(c[1]), axis=1), int(c[2]), axis=2)


class LBM3D:
    def __init__(self, nx, ny, nz, *, u_lb, re, char_length,
                 array_module=np, dtype=np.float32):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.u_lb = u_lb
        self.re = re
        self.char_length = char_length
        self.xp = array_module
        self.dtype = dtype
        xp = self.xp

        self.nu = u_lb * char_length / re
        self.omega = 1.0 / (3.0 * self.nu + 0.5)

        self.solid = xp.zeros((nx, ny, nz), dtype=bool)
        self.body = None
        self.measure_force = True
        self._boundary_force = xp.zeros(3, dtype=dtype)

        # Constant inflow equilibrium for the incoming inlet populations
        # (uniform u = (u_lb, 0, 0), rho = 1) -- one scalar per direction.
        self.eq_in = {}
        for i in range(19):
            cu = 3.0 * C3[i, 0] * u_lb
            self.eq_in[i] = float(W3[i] * (1.0 + cu + 0.5 * cu ** 2
                                           - 1.5 * u_lb ** 2))

        # Start from the uniform inflow field.
        u0 = xp.zeros((3, nx, ny, nz), dtype=dtype)
        u0[0] = u_lb
        self.f = self._equilibrium(xp.ones((nx, ny, nz), dtype=dtype), u0)

    # -- geometry -----------------------------------------------------------
    def add_sphere(self, cx, cy, cz, r):
        xp = self.xp
        x = xp.arange(self.nx)[:, None, None]
        y = xp.arange(self.ny)[None, :, None]
        z = xp.arange(self.nz)[None, None, :]
        self.solid |= (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2

    # -- core ---------------------------------------------------------------
    def _equilibrium(self, rho, u):
        usqr = 1.5 * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        feq = self.xp.empty((19, self.nx, self.ny, self.nz), dtype=self.dtype)
        for i in range(19):
            cu = 3.0 * (C3[i, 0] * u[0] + C3[i, 1] * u[1] + C3[i, 2] * u[2])
            feq[i] = rho * W3[i] * (1.0 + cu + 0.5 * cu ** 2 - usqr)
        return feq

    def macroscopic(self):
        rho = self.f.sum(axis=0)
        u = self.xp.zeros((3, self.nx, self.ny, self.nz), dtype=self.dtype)
        for i in range(19):
            if C3[i, 0]:
                u[0] += C3[i, 0] * self.f[i]
            if C3[i, 1]:
                u[1] += C3[i, 1] * self.f[i]
            if C3[i, 2]:
                u[2] += C3[i, 2] * self.f[i]
        u /= rho
        return rho, u

    def step(self):
        # Outlet (+x): zeroth-order outflow for the inbound populations.
        for i in CX_NEG:
            self.f[i, -1] = self.f[i, -2]
        # Inlet (-x): impose the inflow equilibrium for incoming populations.
        for i in CX_POS:
            self.f[i, 0] = self.eq_in[i]

        rho, u = self.macroscopic()
        feq = self._equilibrium(rho, u)
        fout = self.f - self.omega * (self.f - feq)

        if self.measure_force:
            self._boundary_force = self._momentum_exchange_force(fout)
        for i in range(19):                       # bounce-back at solids
            fout[i, self.solid] = self.f[OPP3[i], self.solid]

        for i in range(19):                       # streaming
            self.f[i] = _roll3(self.xp, fout[i], C3[i])

    def _momentum_exchange_force(self, fout):
        body = self.solid if self.body is None else self.body
        force = self.xp.zeros(3, dtype=self.dtype)
        for i in range(19):
            if i == 0:
                continue
            neigh_body = _roll3(self.xp, body, -C3[i])
            link = neigh_body & ~self.solid
            if not bool(link.any()):
                continue
            m = 2.0 * fout[i][link]               # 2 c_i f_i per link
            force[0] += (C3[i, 0] * m).sum()
            force[1] += (C3[i, 1] * m).sum()
            force[2] += (C3[i, 2] * m).sum()
        return force

    @property
    def force(self):
        return self._boundary_force

    def drag_coefficient(self, area):
        """Cd = Fx / (0.5 rho U^2 A) with rho = 1 (lattice)."""
        return float(self._boundary_force[0]) / (0.5 * self.u_lb ** 2 * area)
