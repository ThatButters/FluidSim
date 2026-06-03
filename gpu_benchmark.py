"""
gpu_benchmark -- validate the GPU solver against the CPU reference and measure
the speed-up.

The solver (lbm2d.LBM2D) is backend-agnostic: NumPy on the CPU, CuPy on the GPU,
running the *identical* code. This script:

  1. CORRECTNESS -- runs the same cylinder on both and confirms they agree to
     machine precision (the GPU must reproduce the validated CPU physics).
  2. THROUGHPUT  -- measures MLUPS (million lattice-cell updates per second) on
     a large domain for each backend and reports the speed-up.

Speed is reported with force extraction off (sim.measure_force = False), since
the per-step host-side force checks stall the GPU pipeline and aren't needed on
every step in practice.
"""

from __future__ import annotations

import time
import numpy as np

try:
    import cupy as cp
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp, HAVE_GPU = None, False

from lbm2d import LBM2D


def _sync(xp):
    if xp is cp:
        cp.cuda.Stream.null.synchronize()


def correctness(steps=500):
    print(f"[correctness] cylinder, {steps} steps, CPU vs GPU ...")
    out = {}
    for name, xp in [("CPU", np), ("GPU", cp)]:
        sim = LBM2D(300, 180, u_lb=0.05, re=100.0, char_length=40.0,
                    array_module=xp)
        sim.add_circle(80.0, 90.0, 20.0)
        cd = np.empty(steps)
        for t in range(steps):
            sim.step()
            cd[t] = sim.coefficients()[0]
        out[name] = cd
    diff = np.abs(out["CPU"] - out["GPU"]).max()
    print(f"  CPU Cd[-1]={out['CPU'][-1]:.6f}  GPU Cd[-1]={out['GPU'][-1]:.6f}")
    print(f"  max |CPU - GPU| over {steps} steps = {diff:.2e}")
    print(f"  -> {'MATCH (machine precision)' if diff < 1e-9 else 'MISMATCH'}\n")
    return diff < 1e-9


def mlups(xp, n, steps, warmup=10):
    sim = LBM2D(n, n, u_lb=0.05, re=100.0, char_length=n / 8.0,
                array_module=xp)
    sim.add_circle(n / 4.0, n / 2.0, n / 16.0)
    sim.measure_force = False                 # pure collide-stream throughput
    for _ in range(warmup):                   # JIT warm-up on GPU
        sim.step()
    _sync(xp)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    _sync(xp)
    dt = time.perf_counter() - t0
    return n * n * steps / dt / 1e6, dt / steps * 1e3


def throughput(n=512):
    print(f"[throughput] {n}x{n} domain ({n*n/1e6:.2f}M cells), "
          f"force extraction off")
    cpu_mlups, cpu_ms = mlups(np, n, steps=40)
    print(f"  CPU : {cpu_mlups:8.1f} MLUPS   ({cpu_ms:.1f} ms/step)")
    if HAVE_GPU:
        gpu_mlups, gpu_ms = mlups(cp, n, steps=300)
        print(f"  GPU : {gpu_mlups:8.1f} MLUPS   ({gpu_ms:.2f} ms/step)")
        print(f"  -> speed-up: {gpu_mlups / cpu_mlups:.0f}x\n")
        # 9 float64 populations dominate memory: ~72 bytes/cell (this NumPy-style
        # layout; a packed native kernel reaches ~55 bytes/cell).
        max_cells = 14 * 1e9 / 72                    # ~14 GB usable of 16
        print(f"  est. max domain in ~14GB at this layout: "
              f"{max_cells/1e6:.0f}M cells (~{int(max_cells**0.5)}^2)")
    else:
        print("  GPU : not available")


if __name__ == "__main__":
    print("=" * 56)
    print("  FluidSim GPU port -- correctness + throughput")
    print("=" * 56)
    if not HAVE_GPU:
        print("No CuPy/GPU detected; running CPU only.\n")
    correctness()
    throughput(512)
    if HAVE_GPU:
        throughput(1024)
