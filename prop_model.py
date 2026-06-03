"""
prop_model -- the propeller-mode engine: a spinning STL in a wind tunnel.

Mirrors FlowModel (same renderer-agnostic contract: returns PyVista meshes +
metrics) but the geometry physically sweeps the grid. The validated recipe
(see validate_prop.py):

  * the disk lies in the y-z plane and spins about +x (the flow/thrust axis);
  * a bank of pre-rotated solid masks is cycled at the true rotation rate, so
    the blades sweep the lattice (fresh-cell refill smooths each advance);
  * the CUDA kernel adds the moving-wall velocity at the blade surfaces.

Honest scope: this drives a stable, physical, swirling slipstream -- great for
visualisation and comparing props -- and reports a robust *reacted torque* and
swirl. Absolute *thrust* on a coarsely-voxelised low-Re prop carries the same
low-Re force caveat documented for airfoils (see docs/VALIDATION.md), so it is
shown but flagged.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp
import pyvista as pv
from cupyx.scipy import ndimage as cndi

from lbm3d_cuda import LBM3D_CUDA
from stl_import import (make_prop, write_binary_stl, load_binary_stl,
                        fit_to_grid, voxelize, orient_prop_to_spin_axis,
                        flip_over)
from flow_model import q_criterion_gpu


class PropModel:
    def __init__(self, nx=140, ny=104, nz=104, u_free=0.03, u_tip=0.04,
                 nu=0.05, k_frames=180):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.u_free = u_free                    # axial freestream (advance)
        self.u_tip = u_tip                      # blade-tip speed (sets RPM)
        self.nu = nu
        self.k_frames = k_frames
        self.cy = (ny - 1) / 2.0
        self.cz = (nz - 1) / 2.0
        self.spin_sign = 1.0                    # +1 = ccw (about +x), -1 = cw
        self._tris = None
        self.sim = None
        self.level = 1e-6
        self.steps = 0
        self._angle = 0.0
        self._cur = 0

    # -- model loading ------------------------------------------------------
    @staticmethod
    def _dir_sign(spin):
        """Map a spin direction (``'ccw'``/``'cw'`` or a signed number) to +/-1."""
        if isinstance(spin, (int, float)):
            return 1.0 if spin >= 0 else -1.0
        return -1.0 if str(spin).lower() in ("cw", "clockwise", "-") else 1.0

    def load(self, stl_path=None, auto_orient=True, spin="ccw", flip=False):
        if stl_path is None:
            out = os.path.join(os.path.dirname(__file__), "out")
            os.makedirs(out, exist_ok=True)
            stl_path = os.path.join(out, "test_prop.stl")
            if not os.path.exists(stl_path):
                write_binary_stl(stl_path, make_prop(n_blades=3, radius=1.0,
                                 thickness=0.09, pitch_deg=14.0))
        self._stl_path = stl_path
        self._auto_orient = auto_orient
        self._flip = bool(flip)
        self.spin_sign = self._dir_sign(spin)
        self._tris = load_binary_stl(stl_path)
        # Auto-align the shaft to +x so any prop spins flat instead of tumbling
        # end-over-end (the solver always spins about +x; see _build_bank).
        self.orient_info = None
        if auto_orient:
            self._tris, self.orient_info = orient_prop_to_spin_axis(
                self._tris, flip=self._flip)
            if self.orient_info["reoriented"]:
                s = np.round(self.orient_info["shaft"], 2)
                fl = "  [flipped]" if self.orient_info["flipped"] else ""
                print(f"[prop] auto-oriented: detected shaft {s} -> +x "
                      f"(flatness {self.orient_info['flatness']:.2f}){fl}")
        elif self._flip:                        # manual mode: flip in place
            self._tris = flip_over(self._tris)
        self._rebuild(warmup=1500)
        return int(self.base.sum())

    def _voxelise(self):
        tris = fit_to_grid(self._tris, self.nx, self.ny, self.nz, 0.24)
        self.base = voxelize(tris, self.nx, self.ny, self.nz)
        # tip radius (in the y-z plane) and rotation rate from the tip speed
        yy, zz = np.where(self.base.any(axis=0))
        self.r_tip = float(np.sqrt((yy - self.cy) ** 2
                                   + (zz - self.cz) ** 2).max()) or 1.0
        self.omega_spin = self.spin_sign * self.u_tip / self.r_tip
        self.area = float(np.pi * self.r_tip ** 2) or 1.0
        # frontal (disk) area for a thrust coefficient if ever needed
        self._build_bank()

    def _build_bank(self):
        g = cp.asarray(self.base, dtype=cp.float32)
        self.bank = []
        for k in range(self.k_frames):
            if k == 0:
                r = g
            else:
                r = cndi.rotate(g, 360.0 * k / self.k_frames, axes=(1, 2),
                                reshape=False, order=0)
            self.bank.append((r > 0.5).astype(cp.uint8))
        self._dframe = 360.0 / self.k_frames

    def _rebuild(self, warmup=0):
        self._voxelise()
        self.sim = LBM3D_CUDA(self.nx, self.ny, self.nz, u_lb=self.u_free,
                              re=600.0, char_length=2 * self.r_tip,
                              collision="les")
        self.sim.omega = np.float32(1.0 / (3.0 * self.nu + 0.5))
        self.sim.set_solid(self.base)
        self.sim.set_spin(self.omega_spin, cy=self.cy, cz=self.cz, on=True)
        # Non-reflecting pressure (anti-bounce-back) outlet: the slipstream leaves
        # without the closed-tank pile-up, so the prop runs at static (J=0)
        # without diverging -- and without an absorbing sponge choking the wake.
        # The inlet stays a velocity inlet (anchors the mean-flow reference;
        # two pressure boundaries would leave it unconstrained and drift).
        self.sim.set_open_ends(open_inlet=False, abb_outlet=True)
        self.body = self._image(self.base.astype(np.float32)).contour([0.5])
        self.steps = 0
        self._angle = 0.0
        self._cur = 0
        for _ in range(warmup):
            self.step(1)
        self.level = self._auto_level()

    # -- live controls ------------------------------------------------------
    def set_rpm_fraction(self, frac):
        """Set the tip speed (hence RPM) from a 0..1 slider fraction. Capped to
        keep the tip Mach modest (incompressible LBM)."""
        self.u_tip = float(np.clip(0.012 + 0.04 * frac, 0.012, 0.052))
        self.omega_spin = self.spin_sign * self.u_tip / self.r_tip
        self.sim.set_spin(self.omega_spin, cy=self.cy, cz=self.cz, on=True)

    def set_spin_direction(self, spin):
        """Reverse the spin (``'ccw'``/``'cw'``) live -- no re-mesh needed; just
        flips the sign of the angular rate fed to the visual bank and the
        moving-wall in the solver."""
        self.spin_sign = self._dir_sign(spin)
        self.omega_spin = self.spin_sign * self.u_tip / self.r_tip
        self.sim.set_spin(self.omega_spin, cy=self.cy, cz=self.cz, on=True)

    def set_flip(self, flip):
        """Flip the prop over (180 deg about an in-plane axis) and re-mesh, for
        when it loaded upside-down. Re-meshing is required because the swept
        mask bank is rebuilt from the new geometry."""
        if bool(flip) == getattr(self, "_flip", False):
            return
        self.load(self._stl_path, auto_orient=self._auto_orient,
                  spin=("cw" if self.spin_sign < 0 else "ccw"), flip=bool(flip))

    def set_wind_fraction(self, frac):
        """Set the flight speed (axial advance) from a 0..1 fraction. ``frac=0``
        is **static / hover run-up** (J=0) -- the basic prop test. The resulting
        advance ratio J also depends on the current RPM (J = pi*V/u_tip), so read
        it live; low RPM + high speed pushes into the high-J windmill regime."""
        self.u_free = float(np.clip(0.052 * frac, 0.0, 0.052))
        self.sim.u_lb = self.u_free

    def relevel(self):
        self.level = self._auto_level()

    def reset(self):
        self._rebuild(warmup=600)

    # -- stepping (advances the swept mask) ---------------------------------
    def step(self, n):
        for _ in range(n):
            self._angle += np.degrees(self.omega_spin)
            nxt = int(self._angle / self._dframe) % self.k_frames
            if nxt != self._cur:
                self.sim.update_solid(self.bank[nxt])
                self._cur = nxt
            self.sim.step()
        self.steps += n

    @property
    def blade_angle(self):
        """Current rotation angle (degrees) -- drives the visual spin."""
        return self._angle % 360.0

    # -- outputs ------------------------------------------------------------
    def _image(self, field):
        grid = pv.ImageData(dimensions=(self.nx, self.ny, self.nz))
        grid.point_data["v"] = field.ravel(order="F")
        return grid

    def _q_field(self):
        _, u = self.sim.macroscopic()
        q = q_criterion_gpu(u)
        q[self.base] = 0.0
        return q

    def _auto_level(self):
        q = self._q_field()
        pos = q[q > 0]
        return float(np.percentile(pos, 92)) if pos.size else 1e-6

    def vortex_mesh(self):
        return self._image(self._q_field()).contour([self.level], scalars="v")

    def metrics(self):
        """Propeller performance numbers from the flow field, at the current
        operating point (RPM + flight speed).

        Robust (wall-carried angular momentum): reacted ``torque`` and the shaft
        ``power`` it implies, ``swirl``, and the non-dimensional power
        coefficient ``c_p``. Estimated, and carrying the documented low-Re voxel
        force caveat: ``thrust`` (an actuator-disk momentum balance), its
        coefficient ``c_t`` and efficiency ``eta`` -- trustworthy for *comparing*
        props and reading curve shapes, not as absolute certified numbers.

        Coefficients use the standard propeller non-dimensionalisation with
        n = rev/step and D = 2*r_tip (lattice units; consistent across runs)::

            J = V/(nD)      C_T = T/(rho n^2 D^4)
            C_P = P/(rho n^3 D^5)      eta = J*C_T/C_P = T*V/P
        """
        rho, u = self.sim.macroscopic()
        ux, uy, uz = u[0], u[1], u[2]
        Y, Z = cp.mgrid[0:self.ny, 0:self.nz]
        dy, dz = (Y - self.cy), (Z - self.cz)
        disk = (dy ** 2 + dz ** 2) <= self.r_tip ** 2

        xdisk = int(np.argmax(self.base.sum(axis=(1, 2))))
        xb = min(xdisk + max(8, int(self.r_tip * 0.5)), self.nx - 3)
        xa = max(xdisk - max(8, int(self.r_tip * 0.5)), 2)

        swb = cp.abs(uy[xb]) + cp.abs(uz[xb])
        swirl = float(cp.mean(swb[disk]))
        lx = dy * uz[xb] - dz * uy[xb]                 # (r x u)_x in the wake
        torque = float(cp.sum((rho[xb] * ux[xb] * lx)[disk]))
        power = torque * float(self.omega_spin)
        slip = float(cp.mean(ux[xb][disk]) - cp.mean(ux[xa][disk]))
        # Thrust: net axial momentum added to the slipstream (actuator-disk
        # balance). T = sum rho*ux*(ux - V) over the wake disk. At V=0 this is
        # the static thrust. Signed: sign = tractor vs pusher along +x.
        thrust = float(cp.sum((rho[xb] * ux[xb] * (ux[xb] - self.u_free))[disk]))

        tip_mach = self.u_tip / 0.5773502692         # cs = 1/sqrt(3)
        # advance ratio J = V / (n D); with n D = u_tip / pi -> J = pi V / u_tip
        advance_j = np.pi * self.u_free / max(self.u_tip, 1e-9)
        # non-dimensional coefficients (rho0 = 1 lattice units). C_T is signed:
        # it crosses zero into the windmill regime at high J (real prop physics).
        # C_P uses |power| (always absorbed). Efficiency counts only *propelling*
        # thrust (T>0); at static (V=0) or windmilling (T<=0) it is 0 by
        # definition, so it never goes >100%.
        n = abs(float(self.omega_spin)) / (2.0 * np.pi)   # rev per step
        D = 2.0 * self.r_tip
        p_mag = abs(power)
        c_t = thrust / max(n ** 2 * D ** 4, 1e-12)
        c_p = p_mag / max(n ** 3 * D ** 5, 1e-12)
        eta = (thrust * self.u_free / p_mag) if (thrust > 0 and p_mag > 1e-12) else 0.0
        # Hover / lift efficiency -- the relevant numbers for a lifting rotor at
        # static (where the cruise efficiency eta is 0 by definition):
        #   * figure of merit FM = ideal induced power / actual shaft power, with
        #     the momentum-theory ideal P_ideal = T^1.5 / sqrt(2 rho A);
        #   * lift_per_power = T / P (thrust per watt -- what drone builders tune).
        # Both use the estimated thrust, so they carry the low-Re force caveat;
        # FM is in (0,1) for a real prop (good ones 0.5-0.8). Defined for T>0.
        A_disk = np.pi * self.r_tip ** 2
        if thrust > 0 and p_mag > 1e-12:
            p_ideal = thrust ** 1.5 / np.sqrt(2.0 * 1.0 * A_disk)
            fom = p_ideal / p_mag
            lift_per_power = thrust / p_mag
        else:
            fom = 0.0
            lift_per_power = 0.0
        return {
            "tip_mach": tip_mach,
            "advance_j": advance_j,
            "swirl_pct": 100.0 * swirl / max(self.u_tip, 1e-9),
            "torque": torque,
            "power": power,
            "slipstream": slip,
            "thrust": thrust,
            "c_t": c_t,
            "c_p": c_p,
            "eta": eta,
            "fom": fom,
            "lift_per_power": lift_per_power,
        }

    def sweep_j(self, n_points=7, settle=4000, progress=None):
        """Performance map: hold the current RPM, step flight speed from static
        (J=0) up across the envelope, let each point settle, and record J vs
        thrust, shaft power, C_T, C_P and efficiency. Returns a dict of equal-
        length lists. The robust curves are power/C_P (and torque); thrust/C_T/
        eta carry the low-Re force caveat -- read them as relative/shape.

        ``progress(i, n, metrics)`` is called after each settled point (for a
        GUI status line). Restores the prior flight speed when done.
        """
        saved = self.u_free
        res = {k: [] for k in ("j", "thrust", "power", "c_t", "c_p", "eta")}
        for i, frac in enumerate(np.linspace(0.0, 1.0, n_points)):
            self.set_wind_fraction(float(frac))
            for _ in range(settle):
                self.step(1)
            m = self.metrics()
            res["j"].append(m["advance_j"])
            for k in ("thrust", "power", "c_t", "c_p", "eta"):
                res[k].append(m[k])
            if progress is not None:
                progress(i + 1, n_points, m)
        self.u_free = saved
        self.sim.u_lb = saved
        return res

    def sweep_rpm(self, n_points=7, settle=4000, j_hover=0.0,
                  rpm_lo=0.15, rpm_hi=1.0, progress=None):
        """Hover lift map: hold flight speed at **static** (J=0 by default) and
        step RPM from low to max, recording thrust (lift), shaft power, figure of
        merit and lift-per-power vs tip speed. This is the drone/multirotor
        chart -- how much lift, and how efficiently, across the throttle range.

        Static hover is stable here because the solver runs an outlet sponge
        (see LBM3D_CUDA.set_sponge, enabled in _rebuild): the pumped slipstream
        is absorbed at the +x boundary instead of accumulating, so there is no
        closed-tank blow-up. ``j_hover`` adds an optional small co-flow
        (V = j_hover*u_tip/pi) if you want a slight climb instead of pure hover.
        Returns a dict of equal-length lists. Robust: power. Estimated (low-Re
        caveat): thrust, FM, lift-per-power. Restores the prior RPM & flight
        speed. ``progress(i, n, metrics)`` is called after each settled point.
        """
        saved_free, saved_tip = self.u_free, self.u_tip
        res = {k: [] for k in
               ("rpm_frac", "tip_mach", "thrust", "power", "fom", "lift_per_power")}
        diverged_at = None
        for i, frac in enumerate(np.linspace(rpm_lo, rpm_hi, n_points)):
            self.set_rpm_fraction(float(frac))
            self.u_free = float(j_hover * self.u_tip / np.pi)   # 0 = pure static
            self.sim.u_lb = self.u_free
            for _ in range(settle):
                self.step(1)
            if not self.is_finite():               # safety backstop (shouldn't trip)
                diverged_at = float(self.u_tip / 0.5773502692)
                break
            m = self.metrics()
            res["rpm_frac"].append(float(frac))
            res["tip_mach"].append(m["tip_mach"])
            for k in ("thrust", "power", "fom", "lift_per_power"):
                res[k].append(m[k])
            if progress is not None:
                progress(i + 1, n_points, m)
        res["diverged_at_tip_mach"] = diverged_at
        self.u_free = saved_free
        self.sim.u_lb = saved_free
        self.u_tip = saved_tip
        self.omega_spin = self.spin_sign * self.u_tip / self.r_tip
        self.sim.set_spin(self.omega_spin, cy=self.cy, cz=self.cz, on=True)
        if diverged_at is not None:                # field is garbage -> clean rebuild
            self._rebuild(warmup=600)
        return res

    def is_finite(self):
        return bool(cp.all(cp.isfinite(self.sim.f_a)))
