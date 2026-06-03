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
