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
              const int les, const float csmag)
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
        if (sx < 0) {                       // inlet: incoming from -x -> inflow eq
            float cu = 3.f * cx[i] * u_in;
            fi[i] = w[i] * (1.f + cu + 0.5f*cu*cu - 1.5f*u_in*u_in);
        } else if (sx >= nx) {              // outlet: zeroth-order open outflow
            fi[i] = f_in[i*nc + idx];
        } else {
            long nidx = ((long)sx*ny + sy)*nz + sz;
            if (solid[nidx])                // bounce-back off a solid neighbour
                fi[i] = f_in[opp[i]*nc + idx];
            else
                fi[i] = f_in[i*nc + nidx];
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

    def step(self):
        self._kernel((self._blocks,), (self._threads,),
                     (self.f_a, self.f_b, self.solid,
                      np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                      self.omega, np.float32(self.u_lb), self._les,
                      self._csmag))
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
