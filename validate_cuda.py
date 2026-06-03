"""
validate_cuda -- check the fused CUDA kernel against the CuPy reference, and
measure the speed-up.

The CuPy solver (lbm3d.LBM3D) is the validated reference (sphere drag vs
Schiller-Naumann). The fused kernel (lbm3d_cuda.LBM3D_CUDA) must reproduce its
flow field -- both solve the same D3Q19 physics; small differences are expected
from the push (roll) vs pull (gather) streaming/boundary formulation, so we
check the velocity field agrees to a few percent and converges to the same wake.

Then we benchmark MLUPS: CuPy reference vs fused kernel, and the kernel on a
large domain to show the throughput that makes real-time 3D reachable.
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp

from lbm3d import LBM3D
from lbm3d_cuda import LBM3D_CUDA


def _speed(u):
    return cp.asnumpy(cp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))


def correctness(nx=140, ny=88, nz=88, r=12.0, steps=2500):
    print(f"[correctness] sphere {nx}x{ny}x{nz}, {steps} steps, "
          f"CuPy reference vs fused kernel ...")
    ref = LBM3D(nx, ny, nz, u_lb=0.05, re=100.0, char_length=2 * r,
                array_module=cp, dtype=cp.float32)
    ref.measure_force = False
    ref.add_sphere(40, ny / 2, nz / 2, r)
    cu = LBM3D_CUDA(nx, ny, nz, u_lb=0.05, re=100.0, char_length=2 * r)
    cu.add_sphere(40, ny / 2, nz / 2, r)

    for _ in range(steps):
        ref.step()
        cu.step()
    cp.cuda.Stream.null.synchronize()

    sref = _speed(ref.macroscopic()[1])
    scu = _speed(cu.macroscopic()[1])
    fluid = cp.asnumpy(~ref.solid)
    scale = sref[fluid].max()
    rms = np.sqrt(np.mean(((sref - scu)[fluid]) ** 2)) / scale
    mx = np.abs((sref - scu)[fluid]).max() / scale
    print(f"  field agreement: RMS={rms:.2%}, max={mx:.2%} of peak speed")
    print(f"  -> {'MATCH (same physics)' if rms < 0.03 else 'investigate'}\n")


def mlups(make, n_steps, warmup=15):
    sim = make()
    for _ in range(warmup):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    nc = sim.nx * sim.ny * sim.nz
    return nc * n_steps / dt / 1e6, dt / n_steps * 1e3


def throughput():
    for (nx, ny, nz) in [(160, 96, 96), (256, 128, 128)]:
        cells = nx * ny * nz
        print(f"[throughput] {nx}x{ny}x{nz} ({cells/1e6:.1f}M cells)")

        def mk_ref():
            s = LBM3D(nx, ny, nz, u_lb=0.05, re=100.0, char_length=30.0,
                      array_module=cp, dtype=cp.float32)
            s.measure_force = False
            return s

        def mk_cu():
            return LBM3D_CUDA(nx, ny, nz, u_lb=0.05, re=100.0, char_length=30.0)

        r_ml, r_ms = mlups(mk_ref, 60)
        c_ml, c_ms = mlups(mk_cu, 200)
        print(f"  CuPy reference : {r_ml:8.1f} MLUPS  ({r_ms:.2f} ms/step)")
        print(f"  fused kernel   : {c_ml:8.1f} MLUPS  ({c_ms:.2f} ms/step)")
        print(f"  -> kernel speed-up over CuPy: {c_ml / r_ml:.1f}x\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Fused CUDA kernel -- correctness + throughput")
    print("=" * 60)
    correctness()
    throughput()
