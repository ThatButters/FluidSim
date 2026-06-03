"""
fluidsim_gui -- the FluidSim desktop application.

A native (PySide6 + VTK) GUI: an imported model sits in a live GPU wind tunnel
embedded in the window, with vortex-core isosurfaces, an orbit camera, real-time
controls (angle of attack, wind speed) and live aerodynamic readouts.

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


class ModelDragStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Left-drag turns the MODEL in the (fixed) wind; right-drag orbits the
    camera; the wheel zooms. Model rotation previews live and re-meshes on
    release."""

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
        self._drag = True
        self._x0, self._y0 = self.GetInteractor().GetEventPosition()
        self._p0, self._yw0 = self._win.model.pitch, self._win.model.yaw
        self._win.begin_drag()

    def _lr(self, *_):
        if self._drag:
            self._drag = False
            self._win.commit_orientation()

    def _rp(self, *_):
        self.StartRotate()            # camera orbit on the right button

    def _rr(self, *_):
        self.EndRotate()

    def _mv(self, *_):
        if self._drag:
            x, y = self.GetInteractor().GetEventPosition()
            self._win.preview_orientation(self._p0 - (y - self._y0) * 0.3,
                                          self._yw0 + (x - self._x0) * 0.3)
        else:
            self.OnMouseMove()        # right-orbit / hover handled by parent

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

        self.model = FlowModel()
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

    def _panel(self):
        panel = QtWidgets.QWidget(objectName="panel")
        panel.setFixedWidth(320)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(22, 20, 22, 20); v.setSpacing(12)

        v.addWidget(QtWidgets.QLabel("FLUIDSIM", objectName="tag"))
        v.addWidget(QtWidgets.QLabel("GPU Wind Tunnel", objectName="title"))

        load = QtWidgets.QPushButton("  Open model (.stl)", objectName="primary")
        load.clicked.connect(self._open_dialog)
        v.addWidget(load)

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

        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("⏸  Pause")
        self.play_btn.clicked.connect(self._toggle)
        reset = QtWidgets.QPushButton("↺  Reset flow")
        reset.clicked.connect(self._reset)
        recam = QtWidgets.QPushButton("⤢  Reset view")
        recam.clicked.connect(self._reset_view)
        row.addWidget(self.play_btn); row.addWidget(reset); row.addWidget(recam)
        v.addLayout(row)

        v.addWidget(QtWidgets.QLabel("LIVE READOUT", objectName="section"))
        self.m_cl = self._metric("Lift  Cl", v)
        self.m_cd = self._metric("Drag  Cd", v)
        self.m_ld = self._metric("L / D", v)

        v.addStretch(1)
        note = QtWidgets.QLabel(
            "Trustworthy for comparing designs and for flow shape. Absolute "
            "numbers are best for bluff / attached bodies — see the docs.",
            objectName="note")
        note.setWordWrap(True)
        v.addWidget(note)
        return panel

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

    # -- scene --------------------------------------------------------------
    def _draw_static(self):
        self.plotter.clear()
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        # wind-direction arrow (flow is +x), upstream of the model
        arrow = pv.Arrow(start=(-0.18 * nx, 0.5 * ny, 0.12 * nz),
                         direction=(1, 0, 0), scale=0.42 * nx,
                         tip_length=0.22, tip_radius=0.05, shaft_radius=0.018)
        self.plotter.add_mesh(arrow, color="#ffb454", name="wind")
        self.plotter.add_text("WIND  →", position="lower_left",
                              font_size=12, color="#ffb454")
        self._body_actor = self.plotter.add_mesh(
            self.model.body, color="#c9d4e3", smooth_shading=True, name="body")
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
        self.status.showMessage("Loading & warming up the flow …")
        QtWidgets.QApplication.processEvents()
        cells = self.model.load(stl, pitch=self.model.pitch, yaw=self.model.yaw)
        self._draw_static()
        self.status.showMessage(
            f"{os.path.basename(stl) if stl else 'demo wing'} — "
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
            "aerodynamic numbers.\nMIT licensed.")

    # -- loop ---------------------------------------------------------------
    def _tick(self):
        if self.playing and not self._dragging and self.model.sim is not None:
            self.model.step(self.spf)
            self._draw_vortex()
            cl, cd, ld = self.model.coefficients()
            self.m_cl.setText(f"{cl:+.3f}")
            self.m_cd.setText(f"{cd:.3f}")
            self.m_ld.setText(f"{ld:+.1f}")
            self.plotter.render()
            self._frames += 1
            if self._fps_t.elapsed() > 700:
                fps = self._frames * 1000 / self._fps_t.elapsed()
                self.status.showMessage(
                    f"GPU live   |   {self.model.steps} steps   |   "
                    f"{fps:.0f} fps")
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
