"""
fluidsim_gui -- the FluidSim desktop application.

A native (PySide6 + VTK) GUI with two modes, both running a live GPU simulation
embedded in the window with vortex-core isosurfaces, an orbit camera and live
readouts:

  * Wind tunnel -- an imported model held in the wind; turn it with the pitch
    and yaw sliders (or by dragging it) and read lift / drag / L-D.
  * Propeller   -- an imported prop spun up on the spot; set the rotation speed
    and advance airspeed and watch the slipstream, with tip Mach, swirl, the
    reacted torque and the shaft power it implies.

    python fluidsim_gui.py                # built-in demo wing
    python fluidsim_gui.py myplane.stl    # open straight into your model
"""

from __future__ import annotations

import sys
import os

import pyvista as pv
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from flow_model import FlowModel
from prop_model import PropModel


class ModelDragStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Wind-tunnel mode: left-drag turns the MODEL in the (fixed) wind, right-
    drag orbits the camera. Propeller mode: left-drag just orbits (the prop
    spins on its own). The wheel always zooms."""

    def __init__(self, win):
        super().__init__()
        self._win = win
        self._drag = False
        self.AddObserver("LeftButtonPressEvent", self._lp)
        self.AddObserver("LeftButtonReleaseEvent", self._lr)
        self.AddObserver("RightButtonPressEvent", self._rp)
        self.AddObserver("RightButtonReleaseEvent", self._rr)
        self.AddObserver("MouseMoveEvent", self._mv)

    def _lp(self, *_):
        if self._win.mode != "wind":          # prop mode: left-drag orbits
            self.StartRotate()
            return
        self._drag = True
        self._x0, self._y0 = self.GetInteractor().GetEventPosition()
        self._p0, self._yw0 = self._win.model.pitch, self._win.model.yaw
        self._win.begin_drag()

    def _lr(self, *_):
        if self._win.mode != "wind":
            self.EndRotate()
            return
        if self._drag:
            self._drag = False
            self._win.commit_orientation()

    def _rp(self, *_):
        self.StartRotate()            # camera orbit on the right button
        self._right = True

    def _rr(self, *_):
        self.EndRotate()

    def _mv(self, *_):
        if self._drag:
            x, y = self.GetInteractor().GetEventPosition()
            self._win.preview_orientation(self._p0 - (y - self._y0) * 0.3,
                                          self._yw0 + (x - self._x0) * 0.3)
        else:
            self.OnMouseMove()        # orbit / hover handled by parent

# -- palette -----------------------------------------------------------------
BG = "#0b0f17"
PANEL = "#141a26"
CARD = "#1b2333"
ACCENT = "#27c4ff"
TEXT = "#e7eef7"
MUTED = "#8a97ab"

QSS = f"""
* {{ font-family: 'Segoe UI', system-ui; color: {TEXT}; }}
QMainWindow, QWidget#panel {{ background: {BG}; }}
QWidget#panel {{ background: {PANEL}; }}
QLabel#title {{ font-size: 22px; font-weight: 700; color: {TEXT}; }}
QLabel#tag {{ font-size: 11px; color: {ACCENT}; letter-spacing: 2px; }}
QLabel#section {{ font-size: 11px; color: {MUTED}; letter-spacing: 2px;
                  margin-top: 6px; }}
QFrame#card {{ background: {CARD}; border-radius: 12px; }}
QLabel#metricVal {{ font-size: 26px; font-weight: 700; color: {ACCENT};
                    font-family: 'Consolas', monospace; }}
QLabel#metricKey {{ font-size: 11px; color: {MUTED}; }}
QLabel#sliderVal {{ color: {ACCENT}; font-weight: 600; }}
QLabel#note {{ color: {MUTED}; font-size: 11px; }}
QPushButton {{ background: {CARD}; border: 1px solid #2a3550; border-radius: 9px;
               padding: 9px 14px; font-size: 13px; }}
QPushButton:hover {{ border: 1px solid {ACCENT}; color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: #04222e; border: none;
                       font-weight: 700; }}
QPushButton#primary:hover {{ background: #54d4ff; }}
QPushButton#mode {{ background: {PANEL}; border: 1px solid #2a3550;
                    padding: 8px 10px; font-size: 12px; }}
QPushButton#mode:checked {{ background: {ACCENT}; color: #04222e; border: none;
                            font-weight: 700; }}
QSlider::groove:horizontal {{ height: 5px; background: #26314a;
                              border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 16px; height: 16px;
                              margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QMenuBar, QMenu {{ background: {PANEL}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {CARD};
                                                 color: {ACCENT}; }}
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, stl=None):
        super().__init__()
        self.setWindowTitle("FluidSim — GPU Wind Tunnel")
        self.resize(1320, 820)
        self.setStyleSheet(QSS)

        self.mode = "wind"
        self.models = {"wind": FlowModel(), "prop": None}
        self.model = self.models["wind"]
        self._current_stl = None
        self.playing = True
        self.spf = 8
        self._dragging = False
        self._preview = (self.model.pitch, self.model.yaw)
        self._body_actor = None
        self._vortex_actor = None
        self._fps_t = QtCore.QElapsedTimer(); self._fps_t.start()
        self._frames = 0

        self._build_ui()
        self._open(stl, initial=True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    # -- layout -------------------------------------------------------------
    def _build_ui(self):
        self._menu()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        self.plotter = QtInteractor(central)
        self.plotter.set_background(BG)
        lay.addWidget(self.plotter.interactor, 1)
        self._install_drag_style()

        lay.addWidget(self._panel())
        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.status.showMessage("Ready")
        self._sync_mode_ui()

    def _panel(self):
        panel = QtWidgets.QWidget(objectName="panel")
        panel.setFixedWidth(320)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(22, 20, 22, 20); v.setSpacing(12)

        v.addWidget(QtWidgets.QLabel("FLUIDSIM", objectName="tag"))
        self.title_lbl = QtWidgets.QLabel("GPU Wind Tunnel", objectName="title")
        v.addWidget(self.title_lbl)

        # mode toggle ------------------------------------------------------
        mrow = QtWidgets.QHBoxLayout(); mrow.setSpacing(8)
        self.mode_btns = {}
        for key, label in (("wind", "🌬  Wind tunnel"), ("prop", "🌀  Propeller")):
            b = QtWidgets.QPushButton(label, objectName="mode")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            mrow.addWidget(b)
            self.mode_btns[key] = b
        self.mode_btns["wind"].setChecked(True)
        v.addLayout(mrow)

        load = QtWidgets.QPushButton("  Open model (.stl)", objectName="primary")
        load.clicked.connect(self._open_dialog)
        v.addWidget(load)

        v.addWidget(self._wind_box())
        v.addWidget(self._prop_box())

        # shared transport controls ----------------------------------------
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("⏸  Pause")
        self.play_btn.clicked.connect(self._toggle)
        reset = QtWidgets.QPushButton("↺  Reset flow")
        reset.clicked.connect(self._reset)
        recam = QtWidgets.QPushButton("⤢  Reset view")
        recam.clicked.connect(self._reset_view)
        row.addWidget(self.play_btn); row.addWidget(reset); row.addWidget(recam)
        v.addLayout(row)

        v.addStretch(1)
        self.note = QtWidgets.QLabel(objectName="note")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        return panel

    def _wind_box(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        v.addWidget(QtWidgets.QLabel("ORIENT THE MODEL IN THE WIND",
                                     objectName="section"))
        self.pitch_lbl = QtWidgets.QLabel(objectName="sliderVal")
        self.pitch = self._slider(-15, 18, int(self.model.pitch), v,
                                  "Pitch  (angle of attack)", self.pitch_lbl, "°")
        self.pitch.valueChanged.connect(
            lambda x: self.pitch_lbl.setText(f"{x:+d}°"))
        self.pitch.sliderReleased.connect(self._on_pitch)

        self.yaw_lbl = QtWidgets.QLabel(objectName="sliderVal")
        self.yaw = self._slider(-45, 45, int(self.model.yaw), v,
                                "Yaw  (turn left / right)", self.yaw_lbl, "°")
        self.yaw.valueChanged.connect(
            lambda x: self.yaw_lbl.setText(f"{x:+d}°"))
        self.yaw.sliderReleased.connect(self._on_yaw)

        v.addWidget(QtWidgets.QLabel("WIND", objectName="section"))
        self.wind_lbl = QtWidgets.QLabel(objectName="sliderVal")
        self.wind = self._slider(1, 10, 5, v, "Wind speed", self.wind_lbl, "")
        self.wind_lbl.setText("5 / 10")
        self.wind.valueChanged.connect(self._on_wind)

        v.addWidget(QtWidgets.QLabel("LIVE READOUT", objectName="section"))
        self.m_cl = self._metric("Lift  Cl", v)
        self.m_cd = self._metric("Drag  Cd", v)
        self.m_ld = self._metric("L / D", v)
        self.wind_box_w = box
        return box

    def _prop_box(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        v.addWidget(QtWidgets.QLabel("SPIN THE PROP", objectName="section"))
        self.rpm_lbl = QtWidgets.QLabel(objectName="sliderVal")
        self.rpm = self._slider(1, 10, 5, v, "Rotation speed", self.rpm_lbl, "")
        self.rpm_lbl.setText("5 / 10")
        self.rpm.valueChanged.connect(self._on_rpm)

        self.adv_lbl = QtWidgets.QLabel(objectName="sliderVal")
        self.adv = self._slider(0, 10, 5, v, "Flight speed  (0 = static)",
                                self.adv_lbl, "")
        self.adv_lbl.setText("5 / 10")
        self.adv.valueChanged.connect(self._on_adv)

        # spin direction (live) + flip-over correction (re-meshes)
        dirrow = QtWidgets.QHBoxLayout()
        dirrow.addWidget(QtWidgets.QLabel("Direction", objectName="metricKey"))
        dirrow.addStretch(1)
        self.dir_btn = QtWidgets.QPushButton("CCW ⟲")
        self.dir_btn.setCheckable(True)
        self.dir_btn.clicked.connect(self._on_dir)
        dirrow.addWidget(self.dir_btn)
        v.addLayout(dirrow)
        self.flip_chk = QtWidgets.QCheckBox("Flip over  (fix upside-down)")
        self.flip_chk.toggled.connect(self._on_flip)
        v.addWidget(self.flip_chk)

        v.addWidget(QtWidgets.QLabel("LIFT  &  EFFICIENCY", objectName="section"))
        self.m_thrust = self._metric("Thrust  (lift, est.)", v)
        self.m_fom = self._metric("Figure of merit  (hover, est.)", v)
        self.m_lpp = self._metric("Lift ÷ power  (est.)", v)
        self.m_power = self._metric("Shaft power", v)

        v.addWidget(QtWidgets.QLabel("OPERATING POINT", objectName="section"))
        self.m_j = self._metric("Advance ratio  J", v)
        self.m_eta = self._metric("Cruise η  (est.)", v)
        self.m_torque = self._metric("Torque  (reacted)", v)
        self.m_swirl = self._metric("Wake swirl  % tip", v)
        self.m_mach = self._metric("Tip Mach", v)

        self.sweep_btn = QtWidgets.QPushButton("⤳  Hover lift sweep  (FM vs RPM)")
        self.sweep_btn.clicked.connect(self._on_sweep)
        v.addWidget(self.sweep_btn)
        cav = QtWidgets.QLabel(
            "Figure of merit = how efficiently the prop turns power into lift "
            "(hover, J≈0). “est.” numbers use the estimated thrust (low-Re voxel "
            "caveat) — trustworthy for comparing props, not as absolute values. "
            "Shaft power / torque are robust.")
        cav.setWordWrap(True); cav.setObjectName("note")
        v.addWidget(cav)
        self.prop_box_w = box
        return box

    def _slider(self, lo, hi, val, parent, label, val_lbl, unit):
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel(label, objectName="metricKey"))
        head.addStretch(1)
        val_lbl.setText(f"{val}{unit}")
        head.addWidget(val_lbl)
        parent.addLayout(head)
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(lo, hi); s.setValue(val)
        parent.addWidget(s)
        return s

    def _metric(self, key, parent):
        card = QtWidgets.QFrame(objectName="card")
        cl = QtWidgets.QHBoxLayout(card)
        cl.setContentsMargins(14, 10, 14, 10)
        cl.addWidget(QtWidgets.QLabel(key, objectName="metricKey"))
        cl.addStretch(1)
        val = QtWidgets.QLabel("—", objectName="metricVal")
        cl.addWidget(val)
        parent.addWidget(card)
        return val

    def _menu(self):
        m = self.menuBar().addMenu("&File")
        op = m.addAction("Open model…"); op.triggered.connect(self._open_dialog)
        m.addSeparator()
        q = m.addAction("Quit"); q.triggered.connect(self.close)
        h = self.menuBar().addMenu("&Help")
        a = h.addAction("About"); a.triggered.connect(self._about)

    # -- mode switching -----------------------------------------------------
    def _set_mode(self, mode):
        if mode == self.mode:
            self._sync_mode_ui()
            return
        self.mode = mode
        if mode == "prop" and self.models["prop"] is None:
            self.models["prop"] = PropModel()
        self.model = self.models[mode]
        self._sync_mode_ui()
        # load the current STL into the now-active model (falls back to its demo)
        self._open(self._current_stl, initial=True)

    def _sync_mode_ui(self):
        for k, b in self.mode_btns.items():
            b.setChecked(k == self.mode)
        self.wind_box_w.setVisible(self.mode == "wind")
        self.prop_box_w.setVisible(self.mode == "prop")
        if self.mode == "wind":
            self.title_lbl.setText("GPU Wind Tunnel")
            self.note.setText(
                "Trustworthy for comparing designs and for flow shape. Absolute "
                "numbers are best for bluff / attached bodies — see the docs.")
        else:
            self.title_lbl.setText("GPU Propeller Test")
            self.note.setText(
                "The blades physically sweep the grid. Swirl and reacted torque "
                "are the robust numbers; absolute thrust on a coarse low-Re prop "
                "carries the documented low-Re caveat — best for visualising and "
                "comparing props.")

    # -- scene --------------------------------------------------------------
    def _draw_static(self):
        self.plotter.clear()
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        # freestream / wind arrow (flow is +x), upstream of the model
        arrow = pv.Arrow(start=(-0.18 * nx, 0.5 * ny, 0.12 * nz),
                         direction=(1, 0, 0), scale=0.42 * nx,
                         tip_length=0.22, tip_radius=0.05, shaft_radius=0.018)
        self.plotter.add_mesh(arrow, color="#ffb454", name="wind")
        label = "WIND  →" if self.mode == "wind" else "AIRFLOW  →"
        self.plotter.add_text(label, position="lower_left",
                              font_size=12, color="#ffb454")
        self._body_actor = self.plotter.add_mesh(
            self.model.body, color="#c9d4e3", smooth_shading=True, name="body")
        if self.mode == "prop":
            self._body_actor.SetOrigin(nx / 2.0, ny / 2.0, nz / 2.0)
        self._draw_vortex()
        self.plotter.camera_position = "yz"
        self.plotter.camera.azimuth = 35
        self.plotter.camera.elevation = 18
        self.plotter.reset_camera()

    def _draw_vortex(self):
        self._vortex_actor = self.plotter.add_mesh(
            self.model.vortex_mesh(), name="vortex", color=ACCENT,
            opacity=0.45, smooth_shading=True)

    def _install_drag_style(self):
        try:
            self._style = ModelDragStyle(self)
            for getter in (lambda: self.plotter.iren.interactor,
                           lambda: self.plotter.render_window.GetInteractor()):
                try:
                    getter().SetInteractorStyle(self._style)
                    return
                except Exception:
                    continue
        except Exception:
            pass            # fall back to default camera controls

    # -- mouse drag = rotate the model in the fixed wind --------------------
    def begin_drag(self):
        self._dragging = True

    def preview_orientation(self, pitch, yaw):
        pitch = max(-18.0, min(18.0, pitch))
        yaw = max(-60.0, min(60.0, yaw))
        self._preview = (pitch, yaw)
        c = (self.model.nx / 2.0, self.model.ny / 2.0, self.model.nz / 2.0)
        for act in (self._body_actor, self._vortex_actor):
            if act is not None:
                try:
                    act.origin = c
                    act.orientation = (0.0, yaw, -pitch)
                except Exception:
                    pass
        self.pitch.blockSignals(True); self.yaw.blockSignals(True)
        self.pitch.setValue(int(round(pitch))); self.yaw.setValue(int(round(yaw)))
        self.pitch.blockSignals(False); self.yaw.blockSignals(False)
        self.pitch_lbl.setText(f"{int(round(pitch)):+d}°")
        self.yaw_lbl.setText(f"{int(round(yaw)):+d}°")
        self.plotter.render()

    def commit_orientation(self):
        self._dragging = False
        pitch, yaw = self._preview
        self.status.showMessage("Re-meshing at new orientation …")
        QtWidgets.QApplication.processEvents()
        self.model.set_orientation(pitch, yaw)
        self._draw_static()
        self.status.showMessage("GPU live")

    # -- actions ------------------------------------------------------------
    def _open(self, stl, initial=False):
        self._current_stl = stl
        self.status.showMessage("Loading & warming up the flow …")
        QtWidgets.QApplication.processEvents()
        if self.mode == "wind":
            cells = self.model.load(stl, pitch=self.model.pitch,
                                    yaw=self.model.yaw)
            kind = "demo wing"
        else:
            spin = "cw" if getattr(self, "dir_btn", None) and \
                self.dir_btn.isChecked() else "ccw"
            flip = bool(getattr(self, "flip_chk", None) and
                        self.flip_chk.isChecked())
            cells = self.model.load(stl, spin=spin, flip=flip)
            kind = "demo prop"
        self._draw_static()
        self.status.showMessage(
            f"{os.path.basename(stl) if stl else kind} — "
            f"{cells} cells   |   GPU live")

    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open STL model", "", "STL meshes (*.stl)")
        if path:
            self._open(path)

    def _on_pitch(self):
        self.status.showMessage("Re-meshing at new pitch …")
        QtWidgets.QApplication.processEvents()
        self.model.set_pitch(self.pitch.value())
        self._draw_static()
        self.status.showMessage("GPU live")

    def _on_yaw(self):
        self.status.showMessage("Re-meshing at new yaw …")
        QtWidgets.QApplication.processEvents()
        self.model.set_yaw(self.yaw.value())
        self._draw_static()
        self.status.showMessage("GPU live")

    def _on_wind(self, level):
        self.wind_lbl.setText(f"{level} / 10")
        if self.model.sim is not None:
            re = 200 + (level - 1) * (800 / 9)        # speed 1..10 -> Re 200..1000
            self.model.set_reynolds(re)
            self.model.relevel()                       # so the wake change shows

    def _on_rpm(self, level):
        self.rpm_lbl.setText(f"{level} / 10")
        if self.model.sim is not None:
            self.model.set_rpm_fraction((level - 1) / 9.0)

    def _on_adv(self, level):
        self.adv_lbl.setText("static" if level == 0 else f"{level} / 10")
        if self.model.sim is not None:
            self.model.set_wind_fraction(level / 10.0)

    def _on_dir(self, checked):
        self.dir_btn.setText("CW ⟳" if checked else "CCW ⟲")
        if self.mode == "prop" and self.model.sim is not None:
            self.model.set_spin_direction("cw" if checked else "ccw")

    def _on_flip(self, checked):
        if self.mode != "prop" or self.model.sim is None:
            return
        self.status.showMessage("Flipping the prop & re-meshing …")
        QtWidgets.QApplication.processEvents()
        self.model.set_flip(checked)
        self._draw_static()
        self.status.showMessage("GPU live")

    def _on_sweep(self):
        """Run a static (hover) RPM sweep and save/open the lift map: thrust and
        figure of merit vs RPM -- how much lift, and how efficiently, across the
        throttle range. Blocks playback while it runs (each point must settle)."""
        if self.mode != "prop" or self.model.sim is None:
            return
        was_playing = self.playing
        self.playing = False
        self.sweep_btn.setEnabled(False)
        try:
            def prog(i, n, m):
                self.status.showMessage(
                    f"Hover sweep … point {i}/{n}   tipMach={m['tip_mach']:.2f}  "
                    f"thrust={m['thrust']:+.3f}  FM={100*m['fom']:.0f}%")
                QtWidgets.QApplication.processEvents()
            res = self.model.sweep_rpm(n_points=6, settle=3000, progress=prog)
            n_ok = len(res["tip_mach"])
            ceil = res.get("diverged_at_tip_mach")
            png = self._plot_sweep(res)
            if png is None:
                self.status.showMessage(
                    "Hover unstable even at the lowest RPM in this solver — "
                    "use the live FM readout at low RPM instead.")
            else:
                note = (f"  (stable to tip Mach ~{ceil:.2f}; higher RPM at static "
                        "diverges in this incompressible solver)") if ceil else ""
                self.status.showMessage(
                    f"Hover lift map saved ({n_ok} pts) → {png}{note}")
                try:
                    os.startfile(png)              # open in default image viewer
                except Exception:
                    pass
        finally:
            self.sweep_btn.setEnabled(True)
            self.playing = was_playing

    def _plot_sweep(self, res):
        """Hover lift map: thrust & power vs RPM, and lift efficiency
        (figure of merit, lift-per-power) vs RPM."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(os.path.dirname(__file__), "out")
        os.makedirs(out, exist_ok=True)
        x = res["tip_mach"]
        if not x:                                  # diverged before any point
            return None
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(x, res["thrust"], "s-", color="#e76f51", label="thrust / lift (est.)")
        ax[0].plot(x, res["power"], "o-", color="#2a9d8f", label="shaft power (robust)")
        ax[0].set_xlabel("tip Mach  (RPM)"); ax[0].set_ylabel("lattice units")
        ax[0].set_title("Lift & power vs RPM"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(x, [100.0 * f for f in res["fom"]], "^-", color="#264653",
                   label="figure of merit % (est.)")
        ax[1].plot(x, res["lift_per_power"], "d--", color="#e76f51",
                   label="lift ÷ power (est.)")
        ax[1].set_xlabel("tip Mach  (RPM)")
        ax[1].set_title("Lift efficiency vs RPM"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.suptitle("Hover lift map (static, J=0) — "
                     "robust: power · estimated: thrust / FM / lift-per-power")
        fig.tight_layout()
        path = os.path.join(out, "prop_hover_map.png")
        fig.savefig(path, dpi=120); plt.close(fig)
        return path

    def _reset_view(self):
        self.plotter.camera_position = "yz"
        self.plotter.camera.azimuth = 35
        self.plotter.camera.elevation = 18
        self.plotter.reset_camera()
        self.plotter.render()

    def _toggle(self):
        self.playing = not self.playing
        self.play_btn.setText("▶  Play" if not self.playing else "⏸  Pause")

    def _reset(self):
        self.model.reset(); self._draw_static()

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, "FluidSim",
            "FluidSim — a free, open-source GPU wind tunnel for the RC "
            "community.\n\nImport an STL, watch the airflow live, read the "
            "aerodynamic numbers. Wind-tunnel and propeller modes.\nMIT "
            "licensed.")

    # -- loop ---------------------------------------------------------------
    def _tick(self):
        if not (self.playing and not self._dragging and self.model.sim is not None):
            return
        self.model.step(self.spf)
        self._draw_vortex()
        if self.mode == "wind":
            cl, cd, ld = self.model.coefficients()
            self.m_cl.setText(f"{cl:+.3f}")
            self.m_cd.setText(f"{cd:.3f}")
            self.m_ld.setText(f"{ld:+.1f}")
        else:
            if not self.model.is_finite():     # near-static at high RPM diverges
                self.status.showMessage("⚠ flow diverged (near-static at high "
                                        "RPM) — adding airspeed & resetting")
                self.adv.setValue(max(self.adv.value(), 4))   # give slipstream an outlet
                self.model.reset()
                return
            if self._body_actor is not None:
                self._body_actor.SetOrientation(self.model.blade_angle, 0, 0)
            mtr = self.model.metrics()
            static = mtr['advance_j'] < 1e-3
            self.m_thrust.setText(f"{mtr['thrust']:+.3f}")
            self.m_fom.setText("—" if mtr['thrust'] <= 0
                               else f"{100.0 * mtr['fom']:.0f}%")
            self.m_lpp.setText("—" if mtr['thrust'] <= 0
                               else f"{mtr['lift_per_power']:.1f}")
            self.m_power.setText(f"{abs(mtr['power']):.3f}")
            self.m_j.setText("static" if static else f"{mtr['advance_j']:.2f}")
            self.m_eta.setText("—" if static else f"{100.0 * mtr['eta']:.0f}%")
            self.m_torque.setText(f"{mtr['torque']:.2f}")
            self.m_swirl.setText(f"{mtr['swirl_pct']:.0f}%")
            self.m_mach.setText(f"{mtr['tip_mach']:.2f}")
        self.plotter.render()
        self._frames += 1
        if self._fps_t.elapsed() > 700:
            fps = self._frames * 1000 / self._fps_t.elapsed()
            self.status.showMessage(
                f"GPU live   |   {self.model.steps} steps   |   {fps:.0f} fps")
            self._frames = 0; self._fps_t.restart()


def main():
    stl = sys.argv[1] if len(sys.argv) > 1 else None
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(stl)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
