"""
render_prop -- 3D still of a spinning propeller and its slipstream wake.

Loads a prop STL (auto-oriented), spins it in the 3D CUDA solver until the
swirling slipstream develops, then renders the blades plus a Q-criterion
isosurface of the wake (the helical tip-vortex tubes the prop flings downstream).
Headless matplotlib, mirrors render_3d.py. Saves out/prop_wake.png.
"""

from __future__ import annotations

import os
import numpy as np
import cupy as cp
from skimage import measure
from skimage.filters import gaussian

from prop_model import PropModel
from render_3d import q_criterion

OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    # Use the real downloaded prop if it's around, else the built-in test prop.
    real = os.path.expanduser(r"~/Downloads/3.5x2.5x3T CCW.stl")
    stl = real if os.path.exists(real) else None

    m = PropModel()
    cells = m.load(stl)                       # auto-orients + warms up
    # A clear, developed slipstream reads best: moderate RPM + a little advance.
    m.set_rpm_fraction(0.75)
    m.set_wind_fraction(0.4)
    print(f"Spinning the prop ({cells} solid cells) to develop the slipstream ...",
          flush=True)
    for _ in range(3000):
        m.step(1)

    mask = cp.asnumpy(m.sim.solid).astype(bool)      # current blade positions
    _, u = m.sim.macroscopic()
    q = q_criterion(cp.asnumpy(u))
    q = gaussian(q, sigma=0.8)                        # light smoothing -> tubes, not blobs
    q[mask] = 0.0
    label = os.path.basename(stl) if stl else "built-in test prop"
    _render(mask, q, label)


def _render(mask, q, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bv, bf, _, _ = measure.marching_cubes(mask.astype(np.float32), 0.5)
    level = np.percentile(q[q > 0], 98.5)            # only the strongest cores -> thin tubes
    vv, vf, _, _ = measure.marching_cubes(q, level, step_size=2)

    fig = plt.figure(figsize=(16, 6.5))
    # flow is +x: a 3/4 view, and an axial view looking down the slipstream
    for n, (elev, azim, title) in enumerate(
            [(18, -60, "3/4 view — blades + swirling slipstream"),
             (4, 2, "axial view — 3-blade tip-vortex helix (+x →)")]):
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        ax.plot_trisurf(vv[:, 0], vv[:, 1], vv[:, 2], triangles=vf,
                        color="#23c0ff", alpha=0.32, linewidth=0,
                        antialiased=False, shade=True)
        ax.plot_trisurf(bv[:, 0], bv[:, 1], bv[:, 2], triangles=bf,
                        color="#e8e8ec", alpha=1.0, linewidth=0, shade=True)
        ax.set_box_aspect(mask.shape)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
    fig.suptitle(f"3D propeller mode: spinning prop + slipstream  "
                 f"({label}, Q-criterion isosurface)")
    fig.tight_layout()
    out = os.path.join(OUT, "prop_wake.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved propeller render -> {out}")


if __name__ == "__main__":
    main()
