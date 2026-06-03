"""
lbm3d_cuda -- native-CUDA fused-kernel D3Q19 solver (the production speed path).

The CuPy backend in lbm3d.py is correct but spends most of its time moving
memory: ~40 full-array passes per step (19 pops x macroscopic/equilibrium/
collide + 19x3 rolls to stream). LBM is memory-bandwidth-bound, so that wastes
the GPU.

This solver does the whole timestep in ONE fused CUDA kernel using the "pull"
scheme: each thread owns one cell, gathers (streams) its 19 populations from the
neighbours -- applying bounce-back where a neighbour is solid and the inlet/
outlet boundary where needed -- then collides and writes once. One read pass,
one write pass. Two ping-pong buffers.

The kernel is plain CUDA C compiled at runtime by CuPy's bundled nvrtc, so it
needs no CUDA toolkit install and runs on Blackwell (sm_120). Validated against
the CuPy reference solver (lbm3d.py) by field comparison; see validate_cuda.py.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

_KERNEL = r'''
extern "C" __global__
void lbm_step(const float* __restrict__ f_in, float* __restrict__ f_out,
              const unsigned char* __restrict__ solid,
              const int nx, const int ny, const int nz,
              const float omega, const float u_in,
              const int les, const float csmag,
              const int spin_on, const float omega_spin,
              const float spin_cy, const float spin_cz,
              const int open_inlet, const int abb_outlet)
{
    const long nc = (long)nx * ny * nz;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc) return;

    // Decode cell coordinates (z fastest -> coalesced).
    int z = idx % nz;
    int y = (idx / nz) % ny;
    int x = idx / ((long)nz * ny);

    // D3Q19 lattice (must match lbm3d.C3 / W3 / OPP3 ordering).
    const int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
    const int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
    const int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
    const int opp[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    const float w[19] = {1.f/3.f,
        1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f};

    // Solid cells: pass through (kept consistent across buffers).
    if (solid[idx]) {
        for (int i = 0; i < 19; ++i) f_out[i*nc + idx] = f_in[i*nc + idx];
        return;
    }

    float fi[19];
    for (int i = 0; i < 19; ++i) {
        int sx = x - cx[i], sy = y - cy[i], sz = z - cz[i];
        // y, z periodic
        if (sy < 0) sy += ny; else if (sy >= ny) sy -= ny;
        if (sz < 0) sz += nz; else if (sz >= nz) sz -= nz;
        if (sx < 0) {                       // inlet
            if (open_inlet) {               // open: anti-bounce-back pressure (rho=1)
                fi[i] = -f_in[opp[i]*nc + idx] + 2.f * w[i];   // lets the prop draw air in
            } else {                        // freestream velocity inflow
                float cu = 3.f * cx[i] * u_in;
                fi[i] = w[i] * (1.f + cu + 0.5f*cu*cu - 1.5f*u_in*u_in);
            }
        } else if (sx >= nx) {              // outlet
            // Non-reflecting pressure outlet (rho=1) only at/near static, where
            // it stops the closed-tank pile-up. With a real freestream the
            // outlet velocity is too high for this low-Mach anti-bounce-back, so
            // fall back to the validated zeroth-order open outflow.
            if (abb_outlet && u_in < 1e-6f) {
                fi[i] = -f_in[opp[i]*nc + idx] + 2.f * w[i];
            } else {
                fi[i] = f_in[i*nc + idx];
            }
        } else {
            long nidx = ((long)sx*ny + sy)*nz + sz;
            if (solid[nidx]) {              // bounce-back off a solid neighbour
                fi[i] = f_in[opp[i]*nc + idx];
                if (spin_on) {
                    // Moving wall: prop spins about the +x (flow) axis at
                    // omega_spin. u_wall = omega_spin x_hat x r, r in the y-z
                    // plane about (spin_cy, spin_cz). Evaluate at the wall cell.
                    float uwy = -omega_spin * ((float)sz - spin_cz);
                    float uwz =  omega_spin * ((float)sy - spin_cy);
                    fi[i] += 6.f * w[i] * (cy[i]*uwy + cz[i]*uwz);
                }
            } else {
                fi[i] = f_in[i*nc + nidx];
            }
        }
    }

    // Macroscopic moments.
    float rho = 0.f, ux = 0.f, uy = 0.f, uz = 0.f;
    for (int i = 0; i < 19; ++i) {
        rho += fi[i];
        ux += cx[i]*fi[i]; uy += cy[i]*fi[i]; uz += cz[i]*fi[i];
    }
    float inv = 1.f / rho;
    ux *= inv; uy *= inv; uz *= inv;
    float usq = 1.5f * (ux*ux + uy*uy + uz*uz);

    if (!les) {
        // BGK collision -> write.
        for (int i = 0; i < 19; ++i) {
            float cu = 3.f * (cx[i]*ux + cy[i]*uy + cz[i]*uz);
            float feq = w[i] * rho * (1.f + cu + 0.5f*cu*cu - usq);
            f_out[i*nc + idx] = fi[i] - omega * (fi[i] - feq);
        }
        return;
    }

    // Regularised collision + Smagorinsky LES (stable to high Reynolds).
    // Equilibrium and the non-equilibrium momentum-flux tensor Pi.
    float feq[19];
    float pxx=0,pyy=0,pzz=0,pxy=0,pxz=0,pyz=0;
    for (int i = 0; i < 19; ++i) {
        float cu = 3.f * (cx[i]*ux + cy[i]*uy + cz[i]*uz);
        feq[i] = w[i] * rho * (1.f + cu + 0.5f*cu*cu - usq);
        float fn = fi[i] - feq[i];
        pxx += cx[i]*cx[i]*fn; pyy += cy[i]*cy[i]*fn; pzz += cz[i]*cz[i]*fn;
        pxy += cx[i]*cy[i]*fn; pxz += cx[i]*cz[i]*fn; pyz += cy[i]*cz[i]*fn;
    }
    // Smagorinsky eddy viscosity -> local relaxation rate.
    float pimag = sqrtf(pxx*pxx + pyy*pyy + pzz*pzz
                        + 2.f*(pxy*pxy + pxz*pxz + pyz*pyz));
    float tau0 = 1.f / omega;
    float taut = 0.5f*(tau0 + sqrtf(tau0*tau0
                       + 25.45584412f*csmag*csmag*pimag/rho));  // 18*sqrt(2)
    float om = 1.f / taut;
    float tr3 = (pxx + pyy + pzz) * (1.f/3.f);
    // Rebuild the regularised non-equilibrium and write.
    for (int i = 0; i < 19; ++i) {
        float cpc = cx[i]*cx[i]*pxx + cy[i]*cy[i]*pyy + cz[i]*cz[i]*pzz
                  + 2.f*(cx[i]*cy[i]*pxy + cx[i]*cz[i]*pxz + cy[i]*cz[i]*pyz);
        float fnreg = 4.5f * w[i] * (cpc - tr3);
        f_out[i*nc + idx] = feq[i] + (1.f - om) * fnreg;
    }
}
'''

C3 = np.array([
    (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
    (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1),
    (-1, 0, -1), (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1),
    (0, -1, 1)], dtype=np.int64)
W3 = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)


class LBM3D_CUDA:
    def __init__(self, nx, ny, nz, *, u_lb, re, char_length,
                 collision="bgk", c_smag=0.16):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.u_lb = u_lb
        self.nu = u_lb * char_length / re
        self.omega = np.float32(1.0 / (3.0 * self.nu + 0.5))
        # "les" = regularised collision + Smagorinsky LES (stable at high Re).
        self._les = np.int32(1 if collision == "les" else 0)
        self._csmag = np.float32(c_smag)
        self.nc = nx * ny * nz
        self._kernel = cp.RawKernel(_KERNEL, "lbm_step")

        # Spinning geometry (propeller mode). Off by default -> a plain wind
        # tunnel. set_spin() turns on the moving-wall bounce-back; rotate the
        # mask each step (update_solid) to physically sweep the blades.
        self._spin_on = np.int32(0)
        self._omega_spin = np.float32(0.0)
        self._spin_cy = np.float32((ny - 1) * 0.5)
        self._spin_cz = np.float32((nz - 1) * 0.5)

        # Open boundaries (default off -> validated velocity-inlet / zeroth-order
        # outlet). set_open_ends() switches to anti-bounce-back pressure
        # boundaries: a non-reflecting outlet and an entrainment inlet, which let
        # a propeller run at static (the prop draws air in and pushes it out
        # without the closed-tank pile-up). See set_open_ends().
        self._open_inlet = np.int32(0)
        self._abb_outlet = np.int32(0)

        # Outlet sponge / absorbing layer (off by default). Near the +x outlet,
        # populations are relaxed toward the far-field equilibrium so the pumped
        # slipstream can drain instead of reflecting and accumulating -- this is
        # what lets a prop run at *static* (zero freestream) without the
        # closed-tank blow-up. Configure with set_sponge().
        self._sponge_start = nx                  # nx == disabled
        self._sponge_strength = 0.0
        self._sponge_kappa = 0.5                 # velocity pull toward far field
        self._sponge_s = None                    # cached ramp (slab, 1, 1)

        self.solid = cp.zeros((nx, ny, nz), dtype=cp.uint8)
        # Two ping-pong buffers, initialised to the uniform inflow equilibrium.
        u0 = np.zeros((3, nx, ny, nz), dtype=np.float32)
        u0[0] = u_lb
        f0 = self._equilibrium_host(np.ones((nx, ny, nz), np.float32), u0)
        self.f_a = cp.asarray(f0)
        self.f_b = cp.empty_like(self.f_a)
        self._threads = 256
        self._blocks = (self.nc + self._threads - 1) // self._threads

    def _equilibrium_host(self, rho, u):
        feq = np.empty((19, self.nx, self.ny, self.nz), np.float32)
        usq = 1.5 * (u[0]**2 + u[1]**2 + u[2]**2)
        for i in range(19):
            cu = 3.0 * (C3[i, 0]*u[0] + C3[i, 1]*u[1] + C3[i, 2]*u[2])
            feq[i] = rho * W3[i] * (1 + cu + 0.5*cu**2 - usq)
        return feq

    def add_sphere(self, cx, cy, cz, r):
        x = cp.arange(self.nx)[:, None, None]
        y = cp.arange(self.ny)[None, :, None]
        z = cp.arange(self.nz)[None, None, :]
        self.solid |= ((x - cx)**2 + (y - cy)**2 + (z - cz)**2
                       < r**2).astype(cp.uint8)

    def set_solid(self, mask):
        """Set the solid mask from a boolean array (e.g. a voxelised STL)."""
        self.solid = cp.asarray(mask, dtype=cp.uint8)

    def set_spin(self, omega_spin, cy=None, cz=None, on=True):
        """Spin the solid about the +x (flow) axis at omega_spin rad/step.

        cy, cz default to the domain centre. This switches on the moving-wall
        bounce-back; call update_solid() with a rotated mask each step to make
        the blades physically sweep the grid."""
        self._spin_on = np.int32(1 if on else 0)
        self._omega_spin = np.float32(omega_spin)
        if cy is not None:
            self._spin_cy = np.float32(cy)
        if cz is not None:
            self._spin_cz = np.float32(cz)

    def set_open_ends(self, open_inlet=False, abb_outlet=True):
        """Switch the streamwise boundaries to anti-bounce-back pressure (rho=1).

        ``abb_outlet`` gives a non-reflecting pressure outlet (the slipstream
        leaves without piling up). ``open_inlet`` makes the inlet permeable too,
        so a static/hovering prop can draw air in through the front -- essential
        for J=0, where a fixed-velocity inlet would starve the disk. Leave both
        off for the validated wind-tunnel cases."""
        self._open_inlet = np.int32(1 if open_inlet else 0)
        self._abb_outlet = np.int32(1 if abb_outlet else 0)

    def set_sponge(self, start_frac=0.75, strength=0.25, vel_relax=0.5):
        """Enable an outlet sponge over the last (1-start_frac) of x.

        Each step the populations in the layer are relaxed toward the
        equilibrium at reference density rho=1 (caps the pressure -- the
        closed-tank instability source) and a velocity pulled a fraction
        ``vel_relax`` of the way from the local velocity toward the far field
        (u_lb, 0, 0). That partial pull bleeds enough momentum to stop the jet
        running away, without the full choke of relaxing all the way to the far
        field. The blend strength ramps linearly from 0 at ``start_frac*nx`` to
        ``strength`` at the outlet. Keep the wake measurement plane upstream of
        ``start_frac*nx``."""
        x0 = int(np.clip(start_frac, 0.0, 0.99) * self.nx)
        self._sponge_start = x0
        self._sponge_strength = float(strength)
        self._sponge_kappa = float(np.clip(vel_relax, 0.0, 1.0))
        if strength <= 0.0 or x0 >= self.nx:
            self._sponge_s = None
            return
        xs = np.arange(self.nx, dtype=np.float32)
        ramp = np.clip((xs - x0) / max(self.nx - 1 - x0, 1), 0.0, 1.0) * strength
        self._sponge_s = cp.asarray(ramp[x0:])[:, None, None]   # (slab, 1, 1)

    def _apply_sponge(self, f):
        """Sponge with *decoupled* pressure and momentum control:

          (1) anchor density by rescaling all populations toward rho=1 (strength
              ``s``) -- this preserves velocity exactly (u = sum c f / sum f is
              scale-invariant), so it caps pressure without choking the jet;
          (2) bleed momentum with a *small* relaxation toward feq(rho, far-field
              velocity) at strength ``s*kappa`` -- just enough to stop the jet
              running away, gentle enough not to choke it.
        """
        x0 = self._sponge_start
        s = self._sponge_s                           # density-anchor ramp (slab,1,1)
        sv = s * self._sponge_kappa                  # small velocity-bleed ramp
        u_in = float(self.u_lb)
        sl = f[:, x0:, :, :]                          # (19, slab, ny, nz)
        # (1) density anchor by rescaling -- velocity untouched
        rho = sl.sum(axis=0)
        sl *= (1.0 + s * (1.0 / rho - 1.0))          # rho -> rho + s*(1-rho)
        # (2) weak momentum bleed toward the far field (u_in, 0, 0)
        rho = sl.sum(axis=0)
        ux = cp.zeros_like(rho); uy = cp.zeros_like(rho); uz = cp.zeros_like(rho)
        for i in range(19):
            if C3[i, 0]:
                ux += C3[i, 0] * sl[i]
            if C3[i, 1]:
                uy += C3[i, 1] * sl[i]
            if C3[i, 2]:
                uz += C3[i, 2] * sl[i]
        inv = 1.0 / rho
        ux *= inv; uy *= inv; uz *= inv
        usq = 1.5 * (ux * ux + uy * uy + uz * uz)
        for i in range(19):
            cu = 3.0 * (float(C3[i, 0]) * ux + float(C3[i, 1]) * uy
                        + float(C3[i, 2]) * uz)
            feq_i = W3[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
            # target far-field equilibrium (same rho), only the velocity differs
            cuf = 3.0 * float(C3[i, 0]) * u_in
            feqf_i = W3[i] * rho * (1.0 + cuf + 0.5 * cuf * cuf - 1.5 * u_in * u_in)
            sl[i] = sl[i] + sv * (feqf_i - feq_i)    # nudge velocity toward far field
        f[:, x0:, :, :] = sl

    def update_solid(self, mask):
        """Swap in a new solid mask (rotated blades), refilling cells that just
        turned from solid to fluid with the equilibrium at the local wall
        velocity -- prevents the void/pressure-spike a bare swap would leave."""
        new = cp.asarray(mask, dtype=cp.uint8)
        fresh = (self.solid != 0) & (new == 0)
        if bool(fresh.any()):
            self._refill(fresh)
        self.solid = new

    def _refill(self, fresh):
        """Set f_a at freshly-uncovered cells to the equilibrium for rho=1 at
        the rigid-body rotation velocity there (smooth re-entry of swept fluid)."""
        ys = cp.arange(self.ny, dtype=cp.float32)[None, :, None]
        zs = cp.arange(self.nz, dtype=cp.float32)[None, None, :]
        uwy = -self._omega_spin * (zs - self._spin_cz)
        uwz = self._omega_spin * (ys - self._spin_cy)
        ux = cp.zeros((self.nx, self.ny, self.nz), dtype=cp.float32)
        uy = cp.broadcast_to(uwy, ux.shape)
        uz = cp.broadcast_to(uwz, ux.shape)
        usq = 1.5 * (ux*ux + uy*uy + uz*uz)
        for i in range(19):
            cu = 3.0 * (float(C3[i, 0])*ux + float(C3[i, 1])*uy
                        + float(C3[i, 2])*uz)
            feq = np.float32(W3[i]) * (1.0 + cu + 0.5*cu*cu - usq)
            self.f_a[i] = cp.where(fresh, feq, self.f_a[i])

    def step(self):
        self._kernel((self._blocks,), (self._threads,),
                     (self.f_a, self.f_b, self.solid,
                      np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                      self.omega, np.float32(self.u_lb), self._les,
                      self._csmag, self._spin_on, self._omega_spin,
                      self._spin_cy, self._spin_cz,
                      self._open_inlet, self._abb_outlet))
        if self._sponge_s is not None:           # absorb the wake at the outlet
            self._apply_sponge(self.f_b)
        self.f_a, self.f_b = self.f_b, self.f_a

    def macroscopic(self):
        f = self.f_a
        rho = f.sum(axis=0)
        u = cp.zeros((3, self.nx, self.ny, self.nz), dtype=cp.float32)
        for i in range(19):
            if C3[i, 0]:
                u[0] += C3[i, 0] * f[i]
            if C3[i, 1]:
                u[1] += C3[i, 1] * f[i]
            if C3[i, 2]:
                u[2] += C3[i, 2] * f[i]
        u /= rho
        return rho, u

    def compute_force(self):
        """Momentum-exchange force [Fx, Fy, Fz] on the solid (lattice units).

        Run periodically, off the hot loop. In the pull scheme the buffer holds
        the post-collision population at each cell, so the validated 2 c_i f_i
        sum over fluid-solid links applies directly (matches the CuPy reference).
        """
        f = self.f_a
        solid = self.solid.astype(bool)
        force = cp.zeros(3, dtype=cp.float64)
        for i in range(1, 19):
            c = C3[i]
            neigh = cp.roll(cp.roll(cp.roll(solid, -int(c[0]), 0),
                                    -int(c[1]), 1), -int(c[2]), 2)
            link = neigh & ~solid
            if not bool(link.any()):
                continue
            m = 2.0 * f[i][link]
            force[0] += float(c[0]) * m.sum()
            force[1] += float(c[1]) * m.sum()
            force[2] += float(c[2]) * m.sum()
        return cp.asnumpy(force)

    def coefficients(self, area):
        """(Cx, Cy, Cz) = 2 F / (rho U^2 A), rho = 1. For flow in +x: Cx is the
        drag coefficient, Cy the lift coefficient."""
        F = self.compute_force()
        return F / (0.5 * self.u_lb ** 2 * area)
