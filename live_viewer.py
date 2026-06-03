"""
live_viewer -- real-time interactive flow, running on the GPU.

A NACA 0012 airfoil sits in a live wind tunnel. Tilt it with the arrow keys and
watch the flow respond -- attached flow, growing wake, separation and stall --
with the lift and drag coefficients updating live. This is the first time the
simulation is something you *watch* rather than a number.

Run it yourself (it opens a window):

    python live_viewer.py                 # GPU if available, else CPU
    python live_viewer.py --cpu           # force the CPU backend
    python live_viewer.py --n 360 220     # smaller/faster domain

Controls:
    Up / Down     increase / decrease angle of attack
    Left / Right  decrease / increase wind speed (Reynolds number)
    f             cycle field:  vorticity -> speed -> pressure
    space         pause / resume
    r             reset the flow
    q  or close   quit

Force extraction is computed once per displayed frame (not every step), so the
live Cl/Cd are real but the GPU pipeline isn't stalled on every sub-step.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from lbm2d import LBM2D, C, W
from demo_airfoil import naca0012_mask


def _backend(force_cpu):
    if force_cpu:
        return np, "CPU (NumPy)"
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            return cp, f"GPU ({name})"
    except Exception:
        pass
    return np, "CPU (NumPy)"


class Viewer:
    FIELDS = ("vorticity", "speed", "pressure")

    def __init__(self, nx, ny, force_cpu):
        self.xp, self.backend = _backend(force_cpu)
        self.nx, self.ny = nx, ny
        self.chord = ny * 0.42
        self.x0, self.y0 = nx * 0.32, ny * 0.5
        self.aoa = 6.0
        self.re = 1200.0
        self.u_lb = 0.05
        self.spf = 10                      # sim steps per displayed frame
        self.field = 0
        self.paused = False
        self.running = True
        self.steps = 0
        self._build()

    # -- simulation setup ---------------------------------------------------
    def _build(self):
        self.sim = LBM2D(self.nx, self.ny, u_lb=self.u_lb, re=self.re,
                         char_length=self.chord, array_module=self.xp)
        self._set_aoa(self.aoa, refill=False)
        self.steps = 0

    def _mask(self, aoa):
        m = naca0012_mask(self.nx, self.ny, self.x0, self.y0, self.chord, aoa)
        return self.sim.xp.asarray(m)

    def _set_aoa(self, aoa, refill=True):
        self.aoa = float(np.clip(aoa, -18.0, 18.0))
        new = self._mask(self.aoa)
        if refill:
            fresh = self.sim.solid & ~new        # cells uncovered by the move
            if bool(fresh.any()):
                u = self.u_lb
                for i in range(9):
                    cu = 3.0 * C[i, 0] * u
                    eq = W[i] * (1.0 + cu + 0.5 * cu ** 2 - 1.5 * u ** 2)
                    self.sim.f[i][fresh] = eq
        self.sim.solid = new
        self.sim.body = new

    def _set_re(self, re):
        self.re = float(np.clip(re, 200.0, 3000.0))
        self.sim.nu = self.u_lb * self.chord / self.re
        self.sim.omega = 1.0 / (3.0 * self.sim.nu + 0.5)

    # -- field for display --------------------------------------------------
    def _field(self):
        if self.FIELDS[self.field] == "vorticity":
            return self.sim.vorticity().T, "RdBu_r", 0.04
        rho, u = self.sim.macroscopic()
        if self.FIELDS[self.field] == "speed":
            s = self.sim.to_host(self.sim.xp.sqrt(u[0] ** 2 + u[1] ** 2))
            s[self.sim.to_host(self.sim.solid)] = np.nan
            return s.T, "viridis", None
        p = self.sim.to_host(rho) / 3.0           # p = c_s^2 rho
        p[self.sim.to_host(self.sim.solid)] = np.nan
        return (p - np.nanmean(p)).T, "coolwarm", 0.01

    # -- input --------------------------------------------------------------
    def on_key(self, e):
        if e.key in ("up",):
            self._set_aoa(self.aoa + 1.0)
        elif e.key == "down":
            self._set_aoa(self.aoa - 1.0)
        elif e.key == "right":
            self._set_re(self.re + 200.0)
        elif e.key == "left":
            self._set_re(self.re - 200.0)
        elif e.key == "f":
            self.field = (self.field + 1) % len(self.FIELDS)
        elif e.key == " ":
            self.paused = not self.paused
        elif e.key == "r":
            self._build()
        elif e.key in ("q", "escape"):
            self.running = False

    def on_close(self, _):
        self.running = False

    # -- main loop ----------------------------------------------------------
    def run(self):
        import matplotlib.pyplot as plt
        plt.rcParams["toolbar"] = "None"
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.canvas.manager.set_window_title("FluidSim -- live wind tunnel")
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        fig.canvas.mpl_connect("close_event", self.on_close)

        data, cmap, lim = self._field()
        kw = dict(vmin=-lim, vmax=lim) if lim else {}
        im = ax.imshow(data, origin="lower", cmap=cmap, animated=True, **kw)
        ax.axis("off")
        hud = ax.text(0.012, 0.975, "", transform=ax.transAxes, va="top",
                      family="monospace", fontsize=10, color="w",
                      bbox=dict(boxstyle="round", fc="black", alpha=0.55))
        help_txt = ("up/down: angle   left/right: wind   f: field   "
                    "space: pause   r: reset   q: quit")
        ax.text(0.5, 0.012, help_txt, transform=ax.transAxes, ha="center",
                family="monospace", fontsize=9, color="w",
                bbox=dict(boxstyle="round", fc="black", alpha=0.45))

        plt.show(block=False)
        t_last, frames = time.perf_counter(), 0
        fps = 0.0
        while self.running:
            if not self.paused:
                for k in range(self.spf):
                    self.sim.measure_force = (k == self.spf - 1)
                    self.sim.step()
                self.steps += self.spf

            data, cmap, lim = self._field()
            im.set_data(data)
            im.set_cmap(cmap)
            if lim:
                im.set_clim(-lim, lim)
            else:
                im.set_clim(0.0, np.nanmax(data) or 1.0)
            cd, cl = self.sim.coefficients()
            frames += 1
            now = time.perf_counter()
            if now - t_last > 0.5:
                fps = frames / (now - t_last)
                t_last, frames = now, 0
            hud.set_text(
                f"{self.backend}\n"
                f"field : {self.FIELDS[self.field]}\n"
                f"AoA   : {self.aoa:+5.1f} deg\n"
                f"Re    : {self.re:6.0f}\n"
                f"Cl    : {cl:+6.3f}\n"
                f"Cd    : {cd:6.3f}\n"
                f"steps : {self.steps:>7d}\n"
                f"fps   : {fps:5.1f}" + ("   [PAUSED]" if self.paused else ""))
            plt.pause(0.001)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="FluidSim live interactive viewer")
    p.add_argument("--cpu", action="store_true", help="force the CPU backend")
    p.add_argument("--n", type=int, nargs=2, default=[440, 280],
                   metavar=("NX", "NY"), help="domain size")
    args = p.parse_args()
    v = Viewer(args.n[0], args.n[1], force_cpu=args.cpu)
    print(f"Launching live viewer on {v.backend}. Close the window or press q "
          f"to quit.")
    v.run()


if __name__ == "__main__":
    main()
