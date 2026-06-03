"""
lbm2d_rotating -- moving/rotating boundary support for the LBM core.

Extends LBM2D with a wall-velocity field so solid surfaces can move (rotate).
This is the foundation of the project's make-or-break feature: spinning
propellers and rotors. We start with the simplest case -- a FIXED annulus
whose inner wall has a tangential velocity (Taylor-Couette) -- to validate the
moving-wall bounce-back against an exact analytical solution, before tackling
geometry that actually sweeps the grid (which additionally needs re-voxelisation).

Moving-wall bounce-back (Ladd): a solid surface moving with velocity u_w injects
momentum into the reflected population,

    f_i_reflected = f_i_incident + 6 w_i rho_w (c_i . u_w)        (c_s^2 = 1/3)

Stationary walls (u_w = 0) reduce to ordinary no-slip bounce-back, so a single
formula handles both the rotating inner wall and the fixed outer wall.
"""

from __future__ import annotations

import numpy as np

from lbm2d import LBM2D, C, W, OPP


class RotatingLBM2D(LBM2D):
    def __init__(self, nx: int, ny: int, **kw) -> None:
        super().__init__(nx, ny, **kw)
        # Wall velocity at each node (non-zero only on moving solid surfaces).
        self.wall_u = np.zeros((2, nx, ny))

    def set_rotation(self, cx: float, cy: float, omega: float,
                     mask: np.ndarray) -> None:
        """Assign solid-body rotation u_w = omega x r to the masked cells.

        cx, cy : rotation centre (lattice units)
        omega  : angular velocity (rad / timestep, +ve = counter-clockwise)
        mask   : boolean array selecting the rotating solid cells
        """
        x = np.arange(self.nx)[:, None] - cx          # (nx, 1)
        y = np.arange(self.ny)[None, :] - cy          # (1, ny)
        # solid-body rotation: u = omega_hat_z x r = omega * (-dy, dx)
        ux = np.broadcast_to(-omega * y, (self.nx, self.ny))
        uy = np.broadcast_to(omega * x, (self.nx, self.ny))
        self.wall_u[0][mask] = ux[mask]
        self.wall_u[1][mask] = uy[mask]

    def _apply_bounceback(self, fout: np.ndarray) -> None:
        """Bounce-back with the moving-wall momentum correction."""
        s = self.solid
        wux, wuy = self.wall_u[0], self.wall_u[1]
        for i in range(9):
            corr = 6.0 * W[i] * (C[i, 0] * wux + C[i, 1] * wuy)
            fout[i, s] = self.f[OPP[i], s] + corr[s]
