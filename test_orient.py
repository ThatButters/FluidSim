"""
test_orient -- lock in the propeller auto-orientation, flip and spin-direction
logic (stl_import.principal_frame / orient_prop_to_spin_axis / flip_over and
PropModel._dir_sign).

CPU-only and dependency-free (numpy + stl_import): no GPU, no solver, no Qt, so
it runs anywhere as a fast regression gate. It builds props in known arbitrary
orientations from the built-in `make_prop` and asserts the detector recovers the
canonical (+x shaft, y-z disk) frame -- so it does NOT depend on any external
STL. If the original download (`~/Downloads/3.5x2.5x3T CCW.stl`) happens to be
present it is checked too, but its absence is not a failure.

Why these invariants matter:
  * the solver always spins about +x with the disk in y-z (see PropModel); a prop
    whose shaft is on another axis tumbles end-over-end unless re-oriented;
  * orientation/flip must be *proper* rotations (det +1) -- a mirror would
    silently invert the prop's handedness (CW<->CCW);
  * the non-prop guard must leave clearly non-flat meshes untouched.

Run:  python test_orient.py    (exit code 0 = all pass, 1 = a check failed)
"""

from __future__ import annotations

import os
import sys
import numpy as np

from stl_import import (make_prop, uv_sphere, load_binary_stl,
                        principal_frame, orient_prop_to_spin_axis, flip_over)


# -- tiny test harness -------------------------------------------------------
_fails = []


def check(cond, msg):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        _fails.append(msg)
    return ok


# -- geometry helpers (rotation-invariant signatures) ------------------------
def tri_areas(t):
    v = t.astype(np.float64).reshape(-1, 3, 3)
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def total_area(t):
    return float(tri_areas(t).sum())


def signed_volume(t):
    v = t.astype(np.float64).reshape(-1, 3, 3)
    return float(np.sum(np.einsum("ni,ni->n", v[:, 0], np.cross(v[:, 1], v[:, 2]))) / 6.0)


def sorted_extents(t):
    """Extents along the mesh's principal axes (ascending) -- invariant under
    any rigid rotation, so a good fingerprint for 'same shape, just turned'.
    Computed in float64 so it isn't tripped by the production float32 transform's
    rounding on large, high-poly meshes."""
    return principal_frame(t.astype(np.float64))[2]


def world_extents(t):
    P = t.reshape(-1, 3)
    return np.array([np.ptp(P[:, k]) for k in range(3)])


def is_rigid(a, b, rtol=2e-3):
    """b is a rigid (distance/area/handedness-preserving) image of a. Relative
    tolerances absorb float32 round-off; a real scale or mirror changes these
    far more than 0.2%, so they're still caught."""
    aa, ab = total_area(a), total_area(b)
    return (np.allclose(sorted_extents(a), sorted_extents(b), rtol=rtol, atol=rtol)
            and abs(aa - ab) <= rtol * max(aa, 1.0)
            and np.sign(signed_volume(a)) == np.sign(signed_volume(b)))


def Rx(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate(t, R):
    return (t.reshape(-1, 3) @ R.T).reshape(t.shape).astype(np.float32)


# -- tests -------------------------------------------------------------------
def test_detect_and_orient():
    """A prop with its shaft on any axis is detected and re-oriented to +x."""
    print("shaft detection + orientation (canonical recovered from any pose)")
    base = make_prop(n_blades=3, radius=1.0, thickness=0.09, pitch_deg=14.0)
    thin_canonical = float(sorted_extents(base)[0])

    # (label, rotation applied to the canonical +x prop)
    poses = {
        "shaft +x (canonical)": np.eye(3),
        "shaft +y (Rz 90)":     Rz(90),
        "shaft +z (Ry 90)":     Ry(90),         # the user's real case
        "tilted (Rx30 Ry40 Rz20)": Rx(30) @ Ry(40) @ Rz(20),
    }
    for label, R in poses.items():
        tris = rotate(base, R)
        known_shaft = R @ np.array([1.0, 0.0, 0.0])     # where +x went
        c, axes, exts = principal_frame(tris)
        det = axes[0]
        check(abs(float(det @ known_shaft)) > 0.999,
              f"{label}: thinnest axis aligns with the true shaft")

        out, info = orient_prop_to_spin_axis(tris)
        we = world_extents(out)
        check(info["reoriented"] and not info["flipped"],
              f"{label}: reoriented (no flip)")
        check(int(np.argmin(we)) == 0 and we[0] < we[1] and we[0] < we[2],
              f"{label}: after orient the thin axis is +x, disk in y-z")
        check(abs(we[0] - thin_canonical) < 1e-2,
              f"{label}: shaft thickness preserved ({we[0]:.3f}~={thin_canonical:.3f})")
        check(is_rigid(tris, out),
              f"{label}: orientation is a proper rotation (no scale, no mirror)")


def test_flip_invariants():
    """orient(flip=True) turns the prop over without mirroring it."""
    print("\nflip via orient (auto path)")
    base = rotate(make_prop(n_blades=3, radius=1.0, thickness=0.09, pitch_deg=14.0),
                  Ry(90))                              # shaft on z, like the download
    a, ia = orient_prop_to_spin_axis(base, flip=False)
    b, ib = orient_prop_to_spin_axis(base, flip=True)
    A, B = a.reshape(-1, 3), b.reshape(-1, 3)

    check(ib["flipped"] and not ia["flipped"], "flip flag set only when flip=True")
    check(np.allclose(world_extents(a), world_extents(b), atol=1e-3),
          "flip keeps extents (it is a rotation, not a rescale)")
    check(np.allclose(B[:, 0], -A[:, 0], atol=1e-4)
          and np.allclose(B[:, 1], -A[:, 1], atol=1e-4)
          and np.allclose(B[:, 2], A[:, 2], atol=1e-4),
          "flip negates x & y, keeps z (180 deg about the shaft-normal)")
    check(np.sign(signed_volume(a)) == np.sign(signed_volume(b)),
          "flip preserves handedness (no mirror -> CW/CCW unchanged)")
    check(int(np.argmin(world_extents(b))) == 0,
          "flipped prop still has shaft on +x, disk in y-z")


def test_flip_over():
    """flip_over (manual path) turns the prop over in place, rigidly."""
    print("\nflip_over (manual path, auto_orient off)")
    base = rotate(make_prop(n_blades=3, radius=1.0, thickness=0.09, pitch_deg=14.0),
                  Rx(20) @ Ry(35))
    f = flip_over(base)
    shaft_before = principal_frame(base)[1][0]
    shaft_after = principal_frame(f)[1][0]
    check(is_rigid(base, f), "flip_over is a proper rotation (rigid, no mirror)")
    check(not np.allclose(base, f, atol=1e-4), "flip_over actually changes the mesh")
    check(abs(float(shaft_before @ shaft_after)) > 0.999,
          "flip_over keeps the shaft on the same axis line (just turns it over)")


def test_nonprop_guard():
    """Clearly non-flat meshes are left untouched."""
    print("\nnon-prop guard (chunky meshes are not reoriented)")
    sphere = uv_sphere(radius=1.0)
    out, info = orient_prop_to_spin_axis(sphere)
    check(not info["reoriented"] and np.array_equal(out, sphere.astype(np.float32)),
          f"sphere left unchanged (flatness {info['flatness']:.2f} > 0.6)")


def test_dir_sign():
    """PropModel._dir_sign maps ccw/cw (and signed numbers) to +/-1.

    Skipped gracefully if prop_model can't import (needs cupy/GPU)."""
    print("\nspin-direction sign mapping")
    try:
        from prop_model import PropModel
    except Exception as e:
        print(f"  [SKIP] prop_model import unavailable ({type(e).__name__}); "
              "dir-sign check skipped on this machine")
        return
    ds = PropModel._dir_sign
    check(ds("ccw") == 1.0 and ds("cw") == -1.0, "'ccw'->+1, 'cw'->-1")
    check(ds("CW") == -1.0 and ds("clockwise") == -1.0, "case/alias insensitive")
    check(ds(0.5) == 1.0 and ds(-2) == -1.0, "signed numbers map by sign")


def test_real_download():
    """Spot-check the actual download if it's still in Downloads."""
    print("\nreal download (optional)")
    path = os.path.expanduser(r"~/Downloads/3.5x2.5x3T CCW.stl")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not present")
        return
    tris = load_binary_stl(path)
    c, axes, exts = principal_frame(tris)
    out, info = orient_prop_to_spin_axis(tris)
    we = world_extents(out)
    check(abs(float(axes[0] @ np.array([0, 0, 1.0]))) > 0.99,
          "3.5x2.5x3T CCW: shaft detected on z")
    check(info["reoriented"] and int(np.argmin(we)) == 0,
          "3.5x2.5x3T CCW: reoriented to +x shaft / y-z disk")
    check(is_rigid(tris, out), "3.5x2.5x3T CCW: orientation is rigid")


def main():
    print("=" * 64)
    print("  PROP ORIENTATION / FLIP / DIRECTION -- regression checks")
    print("=" * 64)
    test_detect_and_orient()
    test_flip_invariants()
    test_flip_over()
    test_nonprop_guard()
    test_dir_sign()
    test_real_download()
    print("=" * 64)
    if _fails:
        print(f"  {len(_fails)} CHECK(S) FAILED:")
        for m in _fails:
            print(f"    - {m}")
        print("=" * 64)
        return 1
    print("  ALL CHECKS PASSED.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
