"""
lbm2d_cuda -- native-CUDA fused-kernel D2Q9 solver (the 2D speed path).

The 2D analogue of lbm3d_cuda.py: the whole timestep (stream + bounce-back +
inlet/outlet + collision) in one fused CUDA kernel via the pull scheme, with two
ping-pong buffers. This is the ~50x speed-up that lets 2D runs afford the
resolution needed for accurate high-Reynolds airfoil drag (the E387 gap).

Supports BGK and the regularised + Smagorinsky-LES collision (les=1), matching
lbm2d.py. Validated against the CuPy reference by field comparison.

D2Q9 ordering matches lbm2d.C (opposite(i) == 8 - i):
    0:( 1, 1) 1:( 1, 0) 2:( 1,-1) 3:( 0, 1) 4:( 0, 0)
    5:( 0,-1) 6:(-1, 1) 7:(-1, 0) 8:(-1,-1)
"""

from __future__ import annotations

import numpy as np
import cupy as cp

_KERNEL = r'''
extern "C" __global__
void lbm2d_step(const float* __restrict__ f_in, float* __restrict__ f_out,
                const unsigned char* __restrict__ solid,
                const int nx, const int ny,
                const float omega, const float u_in,
                const int les, const float csmag,
                const float* __restrict__ gamma, const int gamma_on,
                const float nu_floor)
{
    const long nc = (long)nx * ny;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc) return;
    int y = idx % ny;
    int x = idx / ny;

    const int cx[9] = { 1, 1, 1, 0, 0, 0,-1,-1,-1};
    const int cy[9] = { 1, 0,-1, 1, 0,-1, 1, 0,-1};
    const int opp[9] = {8,7,6,5,4,3,2,1,0};
    const float w[9] = {1.f/36,1.f/9,1.f/36,1.f/9,4.f/9,1.f/9,1.f/36,1.f/9,1.f/36};

    if (solid[idx]) {
        for (int i = 0; i < 9; ++i) f_out[i*nc + idx] = f_in[i*nc + idx];
        return;
    }

    float fi[9];
    for (int i = 0; i < 9; ++i) {
        int sx = x - cx[i], sy = y - cy[i];
        if (sy < 0) sy += ny; else if (sy >= ny) sy -= ny;   // y periodic
        if (sx < 0) {                                        // inlet: inflow eq
            float cu = 3.f * cx[i] * u_in;
            fi[i] = w[i] * (1.f + cu + 0.5f*cu*cu - 1.5f*u_in*u_in);
        } else if (sx >= nx) {                               // outlet: outflow
            fi[i] = f_in[i*nc + idx];
        } else {
            long nidx = (long)sx*ny + sy;
            if (solid[nidx]) fi[i] = f_in[opp[i]*nc + idx];  // bounce-back
            else             fi[i] = f_in[i*nc + nidx];
        }
    }

    float rho=0.f, ux=0.f, uy=0.f;
    for (int i = 0; i < 9; ++i) { rho += fi[i]; ux += cx[i]*fi[i]; uy += cy[i]*fi[i]; }
    float inv = 1.f/rho; ux *= inv; uy *= inv;
    float usq = 1.5f*(ux*ux + uy*uy);

    if (!les) {
        for (int i = 0; i < 9; ++i) {
            float cu = 3.f*(cx[i]*ux + cy[i]*uy);
            float feq = w[i]*rho*(1.f + cu + 0.5f*cu*cu - usq);
            f_out[i*nc + idx] = fi[i] - omega*(fi[i] - feq);
        }
        return;
    }

    // Regularised + Smagorinsky LES.
    float feq[9], pxx=0,pxy=0,pyy=0;
    for (int i = 0; i < 9; ++i) {
        float cu = 3.f*(cx[i]*ux + cy[i]*uy);
        feq[i] = w[i]*rho*(1.f + cu + 0.5f*cu*cu - usq);
        float fn = fi[i] - feq[i];
        pxx += cx[i]*cx[i]*fn; pxy += cx[i]*cy[i]*fn; pyy += cy[i]*cy[i]*fn;
    }
    float pimag = sqrtf(pxx*pxx + pyy*pyy + 2.f*pxy*pxy);
    float tau0 = 1.f/omega;
    float taut = 0.5f*(tau0 + sqrtf(tau0*tau0 + 25.45584412f*csmag*csmag*pimag/rho));
    float om;
    if (gamma_on) {
        // Intermittency from the (externally transported) transition field:
        // laminar (gamma~0) -> only a small stability floor; turbulent
        // (gamma~1) -> full Smagorinsky eddy viscosity.
        float nu_smag = (taut - tau0)*(1.f/3.f);
        float nu_eff = nu_floor + gamma[idx]*nu_smag;
        om = 1.f/(tau0 + 3.f*nu_eff);
    } else {
        om = 1.f/taut;
    }
    float tr3 = (pxx + pyy)*(1.f/3.f);
    for (int i = 0; i < 9; ++i) {
        float cpc = cx[i]*cx[i]*pxx + 2.f*cx[i]*cy[i]*pxy + cy[i]*cy[i]*pyy;
        float fnreg = 4.5f*w[i]*(cpc - tr3);
        f_out[i*nc + idx] = feq[i] + (1.f - om)*fnreg;
    }
}
'''

C = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1],
              [-1, 1], [-1, 0], [-1, -1]], dtype=np.int64)
W = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
OPP = np.array([8 - i for i in range(9)])


class LBM2D_CUDA:
    def __init__(self, nx, ny, *, u_lb, re, char_length,
                 collision="bgk", c_smag=0.16, nu_floor=0.0015):
        self.nx, self.ny = nx, ny
        self.u_lb = u_lb
        self.nu = u_lb * char_length / re
        self.char_length = char_length
        self.omega = np.float32(1.0 / (3.0 * self.nu + 0.5))
        # "transition" uses the LES branch but gates the eddy viscosity by an
        # externally supplied (transported) intermittency field self.gamma.
        self._les = np.int32(1 if collision in ("les", "transition") else 0)
        self._gamma_on = np.int32(1 if collision == "transition" else 0)
        self._csmag = np.float32(c_smag)
        self._nu_floor = np.float32(nu_floor)
        self.nc = nx * ny
        self._kernel = cp.RawKernel(_KERNEL, "lbm2d_step")
        self.solid = cp.zeros((nx, ny), dtype=cp.uint8)
        self.gamma = cp.ones((nx, ny), dtype=cp.float32)   # set by the model

        u0 = np.zeros((2, nx, ny), dtype=np.float32)
        u0[0] = u_lb
        self.f_a = cp.asarray(self._equilibrium_host(np.ones((nx, ny), np.float32), u0))
        self.f_b = cp.empty_like(self.f_a)
        self._threads = 256
        self._blocks = (self.nc + self._threads - 1) // self._threads

    def _equilibrium_host(self, rho, u):
        feq = np.empty((9, self.nx, self.ny), np.float32)
        usq = 1.5 * (u[0] ** 2 + u[1] ** 2)
        for i in range(9):
            cu = 3.0 * (C[i, 0] * u[0] + C[i, 1] * u[1])
            feq[i] = rho * W[i] * (1 + cu + 0.5 * cu ** 2 - usq)
        return feq

    def add_circle(self, cx, cy, r):
        x = cp.arange(self.nx)[:, None]
        y = cp.arange(self.ny)[None, :]
        self.solid |= ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2).astype(cp.uint8)

    def set_solid(self, mask):
        self.solid = cp.asarray(mask, dtype=cp.uint8)

    def step(self):
        self._kernel((self._blocks,), (self._threads,),
                     (self.f_a, self.f_b, self.solid,
                      np.int32(self.nx), np.int32(self.ny),
                      self.omega, np.float32(self.u_lb), self._les, self._csmag,
                      self.gamma, self._gamma_on, self._nu_floor))
        self.f_a, self.f_b = self.f_b, self.f_a

    def macroscopic(self):
        f = self.f_a
        rho = f.sum(axis=0)
        u = cp.zeros((2, self.nx, self.ny), dtype=cp.float32)
        for i in range(9):
            if C[i, 0]:
                u[0] += C[i, 0] * f[i]
            if C[i, 1]:
                u[1] += C[i, 1] * f[i]
        u /= rho
        return rho, u

    def compute_force(self):
        """Momentum-exchange force [Fx, Fy] (lattice units), off the hot loop."""
        f = self.f_a
        solid = self.solid.astype(bool)
        force = cp.zeros(2, dtype=cp.float64)
        for i in range(9):
            if i == 4:
                continue
            neigh = cp.roll(cp.roll(solid, -int(C[i, 0]), 0), -int(C[i, 1]), 1)
            link = neigh & ~solid
            if not bool(link.any()):
                continue
            m = 2.0 * f[i][link]
            force[0] += float(C[i, 0]) * m.sum()
            force[1] += float(C[i, 1]) * m.sum()
        return cp.asnumpy(force)

    def coefficients(self):
        F = self.compute_force()
        return F / (0.5 * self.u_lb ** 2 * self.char_length)
