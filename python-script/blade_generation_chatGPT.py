#!/usr/bin/env python3

"""
EXACT STAGEN AXIAL-CONICAL IMPELLER RECONSTRUCTION
===================================================

Input:
    stagen.out

This script uses the ACTUAL blade contours printed by STAGEN:
    COORDINATES GOING ROUND THE BLADE AFTER ANY ROTATION

Contours extracted:
    1. HUB
    2. TIP

The corresponding STAGEN stream surfaces provide:
    XGRID, YGRID, RGRID

The finished blade is generated ONCE, and the remaining blades 
are created via simple coordinate rotations. NO BOOLEAN UNION IS PERFORMED.

Output:
    lvad_single_blade.step
    lvad_hub.step
    lvad_blades.step
    lvad_impeller.step
    lvad_impeller.stl
"""

from pathlib import Path
import re
import numpy as np
import cadquery as cq


# ======================================================================
# SETTINGS
# ======================================================================

STAGEN_FILE = "stagen.out"
N_BLADES = 6
N_SPAN = 9  # 5-7 for quick test, 15-21 for final production geometry

STL_TOLERANCE = 1.0e-5
STL_ANGULAR_TOLERANCE = 0.15

OUT_SINGLE = "lvad_single_blade.step"
OUT_HUB = "lvad_hub.step"
OUT_BLADES = "lvad_blades.step"
OUT_IMPELLER = "lvad_impeller.step"
OUT_STL = "lvad_impeller.stl"


# ======================================================================
# LOAD STAGEN OUTPUT
# ======================================================================

path = Path(STAGEN_FILE)
if not path.exists():
    raise FileNotFoundError(
        f"\nCannot find {STAGEN_FILE}\nCurrent directory: {Path.cwd()}\n"
    )

lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

print("=" * 80)
print("EXACT STAGEN AXIAL-CONICAL IMPELLER RECONSTRUCTION")
print("=" * 80)
print(f"Read {len(lines)} lines from {STAGEN_FILE}\n")


# ======================================================================
# NUMBER PARSER
# ======================================================================

FLOAT_RE = re.compile(
    r"""
    [-+]?
    (?:
        (?:\d+\.\d*)
        |
        (?:\.\d+)
        |
        (?:\d+)
    )
    (?:[Ee][-+]?\d+)?
    """,
    re.VERBOSE,
)


def numbers(line: str) -> list[float]:
    return [float(x) for x in FLOAT_RE.findall(line)]


# ======================================================================
# FIND COMPLETE STAGEN BLADE CONTOURS
# ======================================================================

BLADE_CONTOUR_MARKER = "COORDINATES GOING ROUND THE BLADE AFTER ANY ROTATION"


def extract_blade_contours() -> list[np.ndarray]:
    starts = [i for i, line in enumerate(lines) if BLADE_CONTOUR_MARKER in line]
    print(f"Complete blade contour blocks found: {len(starts)}")

    contours = []
    for block_no, start in enumerate(starts):
        data = []
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if (
                "COORDINATES GOING ROUND THE BLADE AFTER ANY ROTATION" in line
                or "GRID EXPANSION RATIO" in line
                or "AXIAL,TANGENTIAL AND RADIAL" in line
                or "THE BLADE CENTROID" in line
                or "AXIAL & RADIAL COORDINATES" in line
            ):
                break

            v = numbers(line)
            if len(v) < 3:
                continue

            n = v[0]
            if abs(n - round(n)) > 1.0e-8:
                continue

            n = int(round(n))
            if n < 1 or n > 5000:
                continue

            data.append([n, v[1], v[2]])

        if len(data) >= 250:
            contours.append(np.asarray(data, dtype=float))
            print(f"  contour {block_no + 1}: {len(data)} points")

    if len(contours) < 2:
        raise RuntimeError("Could not find the two complete STAGEN blade contours.")

    return contours


blade_contours = extract_blade_contours()
HUB_CONTOUR = blade_contours[0]
TIP_CONTOUR = blade_contours[-1]

print(f"Hub contour points: {len(HUB_CONTOUR)}")
print(f"Tip contour points: {len(TIP_CONTOUR)}")


# ======================================================================
# STREAM SURFACE EXTRACTION
# ======================================================================

STREAM_MARKER = "AXIAL,TANGENTIAL AND RADIAL COORDINATES ON THE STREAM"


def extract_stream_blocks() -> list[np.ndarray]:
    starts = [i for i, line in enumerate(lines) if STREAM_MARKER in line]
    print(f"Stream surface blocks found: {len(starts)}")

    streams = []
    for block_no, start in enumerate(starts):
        data = []
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if (
                "THE BLADE CENTROID" in line
                or "AXIAL & RADIAL COORDINATES" in line
                or "COORDINATES GOING ROUND THE BLADE" in line
                or "GRID EXPANSION RATIO" in line
            ):
                break

            v = numbers(line)
            if len(v) < 4:
                continue

            n = v[0]
            if abs(n - round(n)) > 1.0e-8:
                continue

            n = int(round(n))
            if n < 1 or n > 10000:
                continue

            data.append([n, v[1], v[2], v[3]])

        if len(data) >= 100:
            streams.append(np.asarray(data, dtype=float))
            print(f"  stream {block_no + 1}: {len(data)} points")

    if len(streams) < 2:
        raise RuntimeError("Could not find hub and tip stream surfaces.")

    return streams


streams = extract_stream_blocks()
HUB_STREAM = streams[0]
TIP_STREAM = streams[-1]


def clean_stream(stream: np.ndarray) -> np.ndarray:
    order = np.argsort(stream[:, 1])
    stream = stream[order]
    x = stream[:, 1]
    keep = np.concatenate([[True], np.diff(x) > 1.0e-12])
    return stream[keep]


HUB_STREAM = clean_stream(HUB_STREAM)
TIP_STREAM = clean_stream(TIP_STREAM)


def report_stream(name: str, stream: np.ndarray):
    X, Y, R = stream[:, 1], stream[:, 2], stream[:, 3]
    print(f"\n{name}")
    print(f"  XLE = {X[0]:.10f}")
    print(f"  XTE = {X[-1]:.10f}")
    print(f"  RLE = {R[0]:.10f}")
    print(f"  RTE = {R[-1]:.10f}")
    print(f"  Rmin = {R.min():.10f}")
    print(f"  Rmax = {R.max():.10f}")


report_stream("HUB STREAM", HUB_STREAM)
report_stream("TIP STREAM", TIP_STREAM)


# ======================================================================
# STREAM INTERPOLATION & LIMITS
# ======================================================================

def stream_values(stream: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys, Rs = stream[:, 1], stream[:, 2], stream[:, 3]
    Y = np.interp(X, Xs, Ys)
    R = np.interp(X, Xs, Rs)
    return Y, R


HUB_XLE, HUB_XTE = HUB_STREAM[:, 1].min(), HUB_STREAM[:, 1].max()
TIP_XLE, TIP_XTE = TIP_STREAM[:, 1].min(), TIP_STREAM[:, 1].max()

COMMON_XLE = max(HUB_XLE, TIP_XLE)
COMMON_XTE = min(HUB_XTE, TIP_XTE)

if COMMON_XTE <= COMMON_XLE:
    raise RuntimeError("Hub and tip stream surfaces do not overlap.")

print("\nCOMMON AXIAL DOMAIN")
print(f"  XLE = {COMMON_XLE:.10f}")
print(f"  XTE = {COMMON_XTE:.10f}")
print(f"  L   = {COMMON_XTE - COMMON_XLE:.10f}")


# ======================================================================
# CONTOUR RESAMPLING
# ======================================================================

def contour_parameter(contour: np.ndarray) -> np.ndarray:
    p = contour[:, 1:3]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1.0e-12:
        raise RuntimeError("Degenerate blade contour.")
    return s / s[-1]


def resample_closed_contour(contour: np.ndarray, n: int) -> np.ndarray:
    p = contour[:, 1:3]
    s = contour_parameter(contour)

    if np.linalg.norm(p[-1] - p[0]) > 1.0e-10:
        p = np.vstack([p, p[0]])
        s = contour_parameter(np.column_stack([np.arange(len(p)), p]))

    target = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.interp(target, s, p[:, 0])
    y = np.interp(target, s, p[:, 1])
    return np.column_stack([x, y])


N_CONTOUR = max(len(HUB_CONTOUR), len(TIP_CONTOUR))
HUB_PROFILE = resample_closed_contour(HUB_CONTOUR, N_CONTOUR)
TIP_PROFILE = resample_closed_contour(TIP_CONTOUR, N_CONTOUR)


# ======================================================================
# 3D SURFACE MAPPING
# ======================================================================

def profile_to_3d(profile: np.ndarray, stream: np.ndarray) -> np.ndarray:
    xs, ys = profile[:, 0], profile[:, 1]
    XLE, XTE = stream[:, 1].min(), stream[:, 1].max()
    axial_chord = XTE - XLE

    X = XLE + xs * axial_chord
    YGRID, R = stream_values(stream, X)
    Y = YGRID + ys * axial_chord
    theta = Y / R

    CY = R * np.cos(theta)
    CZ = R * np.sin(theta)
    return np.column_stack([X, CY, CZ])


print("\n" + "=" * 80)
print("MAPPING HUB/TIP BLADE CONTOURS TO STAGEN STREAM SURFACES")
print("=" * 80)

HUB_3D = profile_to_3d(HUB_PROFILE, HUB_STREAM)
TIP_3D = profile_to_3d(TIP_PROFILE, TIP_STREAM)

print("Hub blade mapped.")
print("Tip blade mapped.")

print("\nPHYSICAL BLADE EXTENTS")
print(f"HUB: X = {HUB_3D[:,0].min():.8f} -> {HUB_3D[:,0].max():.8f}")
print(f"TIP: X = {TIP_3D[:,0].min():.8f} -> {TIP_3D[:,0].max():.8f}")


# ======================================================================
# CAD WIRE & BLADE LOFTING
# ======================================================================

def make_span_section(f: float) -> np.ndarray:
    return (1.0 - f) * HUB_3D + f * TIP_3D


def points_to_wire(points: np.ndarray) -> cq.Wire:
    vectors = [cq.Vector(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if np.linalg.norm(points[-1] - points[0]) > 1.0e-9:
        vectors.append(vectors[0])

    edge = cq.Edge.makeSpline(vectors)
    return cq.Wire.assembleEdges([edge])


print("\n" + "=" * 80)
print("BUILDING ONE AXIAL-CONICAL BLADE")
print("=" * 80)

wires = []
for i in range(N_SPAN):
    f = i / (N_SPAN - 1)
    print(f"  section {i + 1:2d}/{N_SPAN}   span={f:.4f}")
    points = make_span_section(f)
    wires.append(points_to_wire(points))

print("\nLofting ONE blade...")
blade_solid = cq.Solid.makeLoft(wires, ruled=False)
blade = cq.Workplane("XY").newObject([blade_solid])
print("Single blade complete.")

print(f"Writing {OUT_SINGLE}...")
cq.exporters.export(blade, OUT_SINGLE)
print(f"Wrote {OUT_SINGLE}")


# ======================================================================
# HUB GENERATION
# ======================================================================

def create_hub() -> cq.Workplane:
    X = np.linspace(HUB_XLE, HUB_XTE, 400)
    R = np.interp(X, HUB_STREAM[:, 1], HUB_STREAM[:, 3])

    points = [cq.Vector(float(x), float(r), 0.0) for x, r in zip(X, R)]
    outer = cq.Edge.makeSpline(points)

    p_axis_te = cq.Vector(float(X[-1]), 0.0, 0.0)
    p_axis_le = cq.Vector(float(X[0]), 0.0, 0.0)

    e_te = cq.Edge.makeLine(points[-1], p_axis_te)
    e_axis = cq.Edge.makeLine(p_axis_te, p_axis_le)
    e_le = cq.Edge.makeLine(p_axis_le, points[0])

    wire = cq.Wire.assembleEdges([outer, e_te, e_axis, e_le])
    face = cq.Face.makeFromWires(wire)

    solid = cq.Solid.revolve(face, 360.0, cq.Vector(0, 0, 0), cq.Vector(1, 0, 0))
    return cq.Workplane("XY").newObject([solid])


print("\n" + "=" * 80)
print("BUILDING HUB")
print("=" * 80)

hub = create_hub()
print("Hub created.")

print(f"Writing {OUT_HUB}...")
cq.exporters.export(hub, OUT_HUB)
print(f"Wrote {OUT_HUB}")


# ======================================================================
# IMPELLER ASSEMBLY & EXPORT
# ======================================================================

print("\n" + "=" * 80)
print("ROTATING BLADES & CREATING IMPELLER")
print("=" * 80)

blade_solids = []
for i in range(N_BLADES):
    angle = 360.0 * i / N_BLADES
    print(f"  blade {i + 1}/{N_BLADES}   {angle:.3f} deg")
    rotated = blade.rotate((0, 0, 0), (1, 0, 0), angle)
    blade_solids.append(rotated.val())

blade_compound_shape = cq.Compound.makeCompound(blade_solids)
blades = cq.Workplane("XY").newObject([blade_compound_shape])

print(f"Writing {OUT_BLADES}...")
cq.exporters.export(blades, OUT_BLADES)
print(f"Wrote {OUT_BLADES}")

# Full impeller compound (No boolean operations performed)
impeller_shape = cq.Compound.makeCompound([hub.val()] + blade_solids)
impeller = cq.Workplane("XY").newObject([impeller_shape])

print(f"Writing {OUT_IMPELLER}...")
cq.exporters.export(impeller, OUT_IMPELLER)
print(f"Wrote {OUT_IMPELLER}")

print("\nWriting STL (Boolean operations: NONE)...")
cq.exporters.export(
    impeller,
    OUT_STL,
    tolerance=STL_TOLERANCE,
    angularTolerance=STL_ANGULAR_TOLERANCE,
)
print(f"Wrote {OUT_STL}")


# ======================================================================
# FINAL REPORT
# ======================================================================

print("\n" + "=" * 80)
print("DONE: EXACT STAGEN HUB/TIP CONTOURS & ROTATIONAL PATTERN")
print("=" * 80)