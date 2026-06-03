"""
stl_import -- read a binary .stl mesh and voxelise it onto the lattice.

This is the front door of the 3D tool: turn an arbitrary triangle mesh (a wing,
a drone frame, a prop) into the solid mask the LBM solver bounces flow off.

Dependency-free (important on bleeding-edge Python): a small binary-STL reader, a
mesh generator for test shapes, and a parity ray-cast voxeliser. The voxeliser is
validated by round-tripping an analytic sphere (mesh -> voxels -> compare to the
exact sphere mask).

Voxelisation (parity ray-cast along +z): for every grid column (i, j), a ray is
shot in +z; it enters/exits the closed surface at sorted triangle crossings, so
cells between successive crossings are inside. This is the standard CPU method; a
GPU version (the production path) does the same parity test massively in parallel.
"""

from __future__ import annotations

import struct
import numpy as np


# -- binary STL I/O ----------------------------------------------------------
def write_binary_stl(path, tris, normals=None):
    """Write triangles (N,3,3) to a binary STL file."""
    n = len(tris)
    if normals is None:
        normals = np.zeros((n, 3), dtype=np.float32)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)                       # header
        fh.write(struct.pack("<I", n))             # triangle count
        for k in range(n):
            fh.write(struct.pack("<3f", *normals[k]))
            for v in tris[k]:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))         # attribute byte count


def load_binary_stl(path):
    """Read a binary STL; return triangle vertices as (N, 3, 3) float32."""
    with open(path, "rb") as fh:
        fh.read(80)
        (n,) = struct.unpack("<I", fh.read(4))
        tris = np.empty((n, 3, 3), dtype=np.float32)
        for k in range(n):
            data = struct.unpack("<12f", fh.read(48))
            fh.read(2)                             # attribute bytes
            tris[k] = np.array(data[3:12]).reshape(3, 3)
    return tris


# -- test mesh ---------------------------------------------------------------
def uv_sphere(radius=1.0, n_lat=24, n_lon=48):
    """A UV-sphere triangle mesh (N,3,3), for testing the voxeliser."""
    lat = np.linspace(0, np.pi, n_lat + 1)
    lon = np.linspace(0, 2 * np.pi, n_lon + 1)
    P = np.empty((n_lat + 1, n_lon + 1, 3))
    P[..., 0] = radius * np.sin(lat)[:, None] * np.cos(lon)[None, :]
    P[..., 1] = radius * np.sin(lat)[:, None] * np.sin(lon)[None, :]
    P[..., 2] = radius * np.cos(lat)[:, None] * np.ones_like(lon)[None, :]
    tris = []
    for a in range(n_lat):
        for b in range(n_lon):
            p00, p01 = P[a, b], P[a, b + 1]
            p10, p11 = P[a + 1, b], P[a + 1, b + 1]
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
    return np.array(tris, dtype=np.float32)


def naca_wing(chord=1.0, span=2.0, thickness=0.12, n=40):
    """A NACA-00xx wing: an airfoil section extruded along span (z).

    Watertight (side walls + end caps), so the parity voxeliser fills it. The
    chord lies along +x (the flow direction), thickness along y (lift), span
    along z. A real RC-style test shape for the import pipeline.
    """
    xc = (1 - np.cos(np.linspace(0, np.pi, n))) / 2.0          # cosine spacing
    yt = 5 * thickness * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                          - 0.3516 * xc ** 2 + 0.2843 * xc ** 3
                          - 0.1015 * xc ** 4)
    upper = np.stack([xc, yt], axis=1)
    lower = np.stack([xc[::-1], -yt[::-1]], axis=1)
    loop = np.vstack([upper, lower[1:-1]]) * chord             # closed outline
    m = len(loop)
    tris = []
    for k in range(m):                                         # side walls
        a, b = loop[k], loop[(k + 1) % m]
        p0, p1 = [a[0], a[1], 0.0], [b[0], b[1], 0.0]
        p2, p3 = [b[0], b[1], span], [a[0], a[1], span]
        tris += [[p0, p1, p2], [p0, p2, p3]]
    for k in range(1, m - 1):                                  # end caps (fan)
        a, b, c = loop[0], loop[k], loop[k + 1]
        tris.append([[a[0], a[1], 0.0], [c[0], c[1], 0.0], [b[0], b[1], 0.0]])
        tris.append([[a[0], a[1], span], [b[0], b[1], span], [c[0], c[1], span]])
    return np.array(tris, dtype=np.float32)


def _box_tris(corners):
    """12 triangles (a closed box) from 8 corners ordered as the cube vertices
    (a,b,c) with a fastest: 000,001,010,011,100,101,110,111."""
    c = corners
    quads = [(0, 2, 3, 1), (4, 5, 7, 6),          # a- / a+ faces
             (0, 1, 5, 4), (2, 6, 7, 3),          # c- / c+ faces
             (0, 4, 6, 2), (1, 3, 7, 5)]          # b- / b+ faces
    tris = []
    for p, q, r, s in quads:
        tris += [[c[p], c[q], c[r]], [c[p], c[r], c[s]]]
    return tris


def make_prop(n_blades=2, radius=1.0, root=0.14, chord=0.30,
              thickness=0.055, pitch_deg=18.0, camber=0.09, n_chord=10):
    """A propeller test mesh: a hub plus `n_blades` cambered, pitched blades.

    The disk lies in the y-z plane so it spins about +x (the flow/thrust axis,
    matching the solver's spin convention). Each blade is a thin *cambered*
    plate (a curved mean line) -- crucial because at the low chord-Reynolds
    numbers of a voxelised model prop, a flat plate is a poor lifting surface
    but a cambered plate generates real thrust. Each blade is pitched about its
    own radial axis. Watertight, so the parity voxeliser fills it.
    """
    phi = np.radians(pitch_deg)
    cp_, sp = np.cos(phi), np.sin(phi)
    t2, c2 = thickness / 2.0, chord / 2.0
    height = camber * chord
    tris = []

    # Hub: a small cube straddling the origin, spanning the spin axis.
    # Corners ordered 000..111 with a(=x) fastest, to match _box_tris.
    h = max(root, thickness)
    hub = [[sx * thickness, sy * h, sz * h]
           for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    tris += _box_tris(np.array(hub, dtype=np.float32))

    bs = np.linspace(-c2, c2, n_chord + 1)        # chordwise stations
    spans = (root, radius)                         # root, tip

    for k in range(n_blades):
        theta = 2 * np.pi * k / n_blades
        ct, st = np.cos(theta), np.sin(theta)

        def xf(a, b, c):
            ap = a * cp_ - b * sp                  # pitch about radial axis
            bp = a * sp + b * cp_
            return [ap, bp * (-st) + c * ct, bp * ct + c * st]

        # top/bottom surface grids: a = mean-line(b) +/- t2
        top, bot = [], []
        for si, cc in enumerate(spans):
            tr, br = [], []
            for b in bs:
                m = height * (1.0 - (b / c2) ** 2)  # parabolic camber
                tr.append(xf(m + t2, b, cc))
                br.append(xf(m - t2, b, cc))
            top.append(tr); bot.append(br)

        def quad(p, q, r, s):
            tris.append([p, q, r]); tris.append([p, r, s])

        for i in range(n_chord):                    # top & bottom sheets
            quad(top[0][i], top[0][i+1], top[1][i+1], top[1][i])
            quad(bot[0][i], bot[1][i], bot[1][i+1], bot[0][i+1])
        for i in range(n_chord):                    # root & tip edges
            quad(top[0][i], top[0][i+1], bot[0][i+1], bot[0][i])
            quad(top[1][i], bot[1][i], bot[1][i+1], top[1][i+1])
        # leading & trailing edge caps
        quad(top[0][0], bot[0][0], bot[1][0], top[1][0])
        quad(top[0][n_chord], top[1][n_chord], bot[1][n_chord], bot[0][n_chord])

    return np.array(tris, dtype=np.float32)


# -- orientation (auto-detect the spin axis) ---------------------------------
def principal_frame(tris):
    """Area-weighted principal axes of a triangle mesh, ordered by ascending
    extent.

    Returns ``(centroid (3,), axes (3,3) rows=unit dirs, extents (3,))``. The
    first axis is the mesh's *thinnest* direction; for a propeller that is the
    shaft -- the disk's normal, i.e. the axis it should spin about. Weighting by
    triangle area (not raw vertices) keeps the result independent of how finely
    each face happens to be tessellated.
    """
    v = tris.reshape(-1, 3, 3)
    c = v.mean(axis=1)                                  # triangle centroids
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    w = area / max(area.sum(), 1e-12)
    centroid = (w[:, None] * c).sum(0)
    d = c - centroid
    cov = np.einsum("n,ni,nj->ij", w, d, d)             # area-weighted covariance
    _, evecs = np.linalg.eigh(cov)
    axes = evecs.T                                      # rows = principal dirs
    P = tris.reshape(-1, 3)
    exts = np.array([np.ptp(P @ axes[k]) for k in range(3)])
    order = np.argsort(exts)                            # ascending extent
    return centroid, axes[order], exts[order]


def orient_prop_to_spin_axis(tris, flatness_max=0.6, flip=False):
    """Rotate a propeller mesh into the solver's canonical frame.

    The solver always spins about +x with the disk in the y-z plane (see
    PropModel). A propeller exported by someone else can have its shaft on any
    axis -- if it does, the solver spins it about the wrong axis and the blades
    tumble end-over-end instead of sweeping a disk. This detects the shaft as
    the mesh's thinnest principal axis and rotates the geometry so the hub sits
    at the origin and the shaft lies along +x.

    The shaft *direction* from PCA is sign-ambiguous, so a prop can come out
    facing the wrong way (effectively upside-down). ``flip=True`` turns it 180
    deg about an in-plane axis -- swapping which face leads -- to correct that.
    The flip is a proper rotation (det +1), so it never changes the prop's
    handedness (CW/CCW); spin direction is a separate control on PropModel.

    Only a clearly flat (disk-like) mesh is touched: if the thinnest/next-axis
    extent ratio is above ``flatness_max`` the mesh isn't prop-like and is
    returned unchanged. Returns ``(oriented_tris (N,3,3) float32, info)`` where
    ``info`` records ``shaft``, ``centroid``, ``extents``, ``flatness``,
    ``reoriented`` and ``flipped``.
    """
    centroid, axes, exts = principal_frame(tris)
    flatness = float(exts[0] / max(exts[1], 1e-12))
    info = {"centroid": centroid, "shaft": axes[0], "extents": exts,
            "flatness": flatness, "reoriented": False, "flipped": False}
    if flatness > flatness_max:                         # not clearly a disk
        return tris.astype(np.float32), info
    x = axes[0] / np.linalg.norm(axes[0])               # shaft -> +x
    y = axes[1] - (axes[1] @ x) * x                     # re-orthonormalise
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if flip:                                            # 180 deg about z: x,y -> -x,-y
        x, y = -x, -y                                   # (proper rotation, det +1)
        info["flipped"] = True
    M = np.column_stack([x, y, z])                      # new = (p - hub) @ M
    out = ((tris.reshape(-1, 3) - centroid) @ M).reshape(tris.shape)
    info["reoriented"] = True
    return out.astype(np.float32), info


def flip_over(tris):
    """Turn a mesh 180 deg about its in-plane axis, in place (keeps its current
    orientation, just swaps which face leads). For fixing an upside-down prop
    when auto-orientation is *off*. Proper rotation, so handedness is preserved.
    """
    centroid, axes, _ = principal_frame(tris)
    a = axes[2]                                         # an in-plane axis
    R = 2.0 * np.outer(a, a) - np.eye(3)               # 180 deg about a
    out = ((tris.reshape(-1, 3) - centroid) @ R.T + centroid).reshape(tris.shape)
    return out.astype(np.float32)


# -- voxelisation ------------------------------------------------------------
def fit_to_grid(tris, nx, ny, nz, margin=0.15):
    """Scale & centre a mesh to fill a fraction of an (nx,ny,nz) grid."""
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    span = np.maximum(hi - lo, 1e-9)
    grid = np.array([nx, ny, nz], dtype=float)
    scale = ((1 - 2 * margin) * grid / span).min()
    centred = (tris - (lo + hi) / 2.0) * scale + grid / 2.0
    return centred


def voxelize(tris, nx, ny, nz):
    """Solid mask (nx,ny,nz) for a closed mesh, by +z parity ray-casting."""
    cols_i, cols_j, cols_z = [], [], []
    for v0, v1, v2 in tris:
        # xy bounding box of the triangle, clipped to the grid columns.
        x0 = int(np.floor(min(v0[0], v1[0], v2[0])))
        x1 = int(np.ceil(max(v0[0], v1[0], v2[0])))
        y0 = int(np.floor(min(v0[1], v1[1], v2[1])))
        y1 = int(np.ceil(max(v0[1], v1[1], v2[1])))
        x0, x1 = max(x0, 0), min(x1, nx - 1)
        y0, y1 = max(y0, 0), min(y1, ny - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1),
                             indexing="ij")
        gx = gx.ravel().astype(float)
        gy = gy.ravel().astype(float)
        # Barycentric coords in xy to test inclusion and interpolate z.
        d = ((v1[1] - v2[1]) * (v0[0] - v2[0])
             + (v2[0] - v1[0]) * (v0[1] - v2[1]))
        if abs(d) < 1e-9:                          # degenerate (edge-on) tri
            continue
        a = ((v1[1] - v2[1]) * (gx - v2[0])
             + (v2[0] - v1[0]) * (gy - v2[1])) / d
        b = ((v2[1] - v0[1]) * (gx - v2[0])
             + (v0[0] - v2[0]) * (gy - v2[1])) / d
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        zc = a * v0[2] + b * v1[2] + c * v2[2]     # z where the ray hits
        cols_i.append(gx[inside].astype(np.int32))
        cols_j.append(gy[inside].astype(np.int32))
        cols_z.append(zc[inside])

    solid = np.zeros((nx, ny, nz), dtype=bool)
    if not cols_i:
        return solid
    ii = np.concatenate(cols_i)
    jj = np.concatenate(cols_j)
    zz = np.concatenate(cols_z)
    # Group crossings by column and parity-fill between sorted pairs.
    key = ii.astype(np.int64) * ny + jj
    order = np.lexsort((zz, key))
    key, zz = key[order], zz[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    for seg in np.split(np.arange(key.size), bounds):
        zlist = np.sort(zz[seg])
        i = int(key[seg[0]] // ny)
        j = int(key[seg[0]] % ny)
        for p in range(0, len(zlist) - 1, 2):      # (enter, exit) pairs
            za = int(np.ceil(zlist[p]))
            zb = int(np.floor(zlist[p + 1]))
            if zb >= za:
                solid[i, j, max(za, 0):min(zb + 1, nz)] = True
    return solid


def load_and_voxelize(path, nx, ny, nz, margin=0.15):
    """Convenience: read an STL and return its fitted solid mask."""
    tris = fit_to_grid(load_binary_stl(path), nx, ny, nz, margin)
    return voxelize(tris, nx, ny, nz)
