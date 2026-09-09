#!/usr/bin/env python3
"""
STAGEN STREAM SURFACE -> PARAMETRIC 6-BLADE IMPELLER GENERATOR

Converts STAGEN stream surface data into a fully parametric 3D solid 6-blade 
impeller mesh with customizable hub, blade thickness, span, and scaling settings.

Outputs:
    - STL (.stl)
    - Wavefront OBJ (.obj)
"""

import math
import re
import sys
from pathlib import Path
import numpy as np

# ============================================================
# USER SETTINGS (IMPELLER & HUB DIMENSIONS)
# ============================================================

INPUT_FILE = "stagen.out"
OUTPUT_STL = "impeller_6blade.stl"
OUTPUT_OBJ = "impeller_6blade.obj"

# --- Blade Parameters ---
N_BLADES = 6          # Number of blades around full wheel (60° spacing)
MAX_THICKNESS = 0.08  # Max profile thickness relative to axial chord (0.08 = 8%)
SPAN_HEIGHT = 0.25    # Radial length/height of the blade span

# --- Hub Cylinder Dimensions ---
INCLUDE_HUB = True    # Set False to export blades only without the hub
CUSTOM_HUB_RADIUS = None  # Set float value (e.g., 0.05) or None to auto-size from STAGEN hub
HUB_AXIAL_EXTENSION_UPSTREAM = 0.01    # Extension distance past leading edge
HUB_AXIAL_EXTENSION_DOWNSTREAM = 0.01  # Extension distance past trailing edge

# --- Mesh & Unit Scaling ---
N_SPAN = 25           # Radial grid resolution along span
MODEL_SCALE = 1.0     # Global multiplier (e.g., 0.001 to convert mm to meters)


# ============================================================
# STAGEN PARSER
# ============================================================

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_stagen_stream_surface(filename):
    """Parse STAGEN stream surface coordinates (X, Y, R)."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {filename}\nExpected path: {path.resolve()}")

    lines = path.read_text(errors="ignore").splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        u = line.upper()
        if "AXIAL" in u and ("TANGENTIAL" in u or "RADIAL" in u) and "STREAM" in u:
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError("Could not locate stream-surface block in stagen.out file.")

    row_pattern = re.compile(
        rf"^\s*(\d+)\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})"
    )

    pts = []
    for line in lines[start_idx + 1:]:
        m = row_pattern.match(line)
        if m:
            pts.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        elif pts and len(line.strip()) == 0:
            break

    if not pts:
        raise RuntimeError("Failed to parse numeric profile data rows from stagen.out.")

    pts = np.array(pts)
    return {"X": pts[:, 0], "Y": pts[:, 1], "R": pts[:, 2]}


# ============================================================
# GEOMETRY GENERATION
# ============================================================

def generate_closed_airfoil(x, y, max_t_ratio):
    """Generates closed suction/pressure side loop around stream line."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12

    nx = -dy / ds
    ny = dx / ds

    chord = x.max() - x.min()
    if chord <= 0:
        chord = 1.0
    t = (x - x.min()) / chord

    t_max = chord * max_t_ratio
    thick = 5.0 * t_max * (
        0.2969 * np.sqrt(np.maximum(t, 0.0))
        - 0.1260 * t
        - 0.3516 * (t ** 2)
        + 0.2843 * (t ** 3)
        - 0.1030 * (t ** 4)
    )

    x_suction = x + 0.5 * thick * nx
    y_suction = y + 0.5 * thick * ny
    x_pressure = x - 0.5 * thick * nx
    y_pressure = y - 0.5 * thick * ny

    x_closed = np.concatenate([x_suction, x_pressure[::-1]])
    y_closed = np.concatenate([y_suction, y_pressure[::-1]])

    return x_closed, y_closed


def build_single_blade(data):
    """Extrudes 2D closed profile into a solid 3D blade."""
    x_cam, y_cam, r_base = data["X"], data["Y"], data["R"]
    r_hub = float(np.mean(r_base))

    x_2d, y_2d = generate_closed_airfoil(x_cam, y_cam, MAX_THICKNESS)
    n_pts = len(x_2d)

    r_stations = np.linspace(r_hub, r_hub + SPAN_HEIGHT, N_SPAN)
    grid = np.zeros((N_SPAN, n_pts, 3))

    for s_idx, r_val in enumerate(r_stations):
        theta = y_2d / r_hub
        grid[s_idx, :, 0] = x_2d
        grid[s_idx, :, 1] = r_val * np.cos(theta)
        grid[s_idx, :, 2] = r_val * np.sin(theta)

    vertices = grid.reshape(-1, 3)

    faces = []
    for s in range(N_SPAN - 1):
        for k in range(n_pts):
            k_next = (k + 1) % n_pts
            p0 = s * n_pts + k
            p1 = s * n_pts + k_next
            p2 = (s + 1) * n_pts + k_next
            p3 = (s + 1) * n_pts + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # Cap hub and tip ends
    for k in range(1, n_pts - 1):
        faces.append([0, k + 1, k])
        top_off = (N_SPAN - 1) * n_pts
        faces.append([top_off, top_off + k, top_off + k + 1])

    return vertices, np.array(faces, dtype=int), r_hub


def build_hub_disk(radius, x_min, x_max, n_theta=64):
    """Generates customizable central shaft hub cylinder."""
    angles = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)

    v_hub_bot = np.column_stack([
        np.full(n_theta, x_min),
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    v_hub_top = np.column_stack([
        np.full(n_theta, x_max),
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])

    vertices = np.vstack([v_hub_bot, v_hub_top])
    faces = []

    for k in range(n_theta):
        k_next = (k + 1) % n_theta
        b0, b1 = k, k_next
        t0, t1 = n_theta + k, n_theta + k_next

        # Side walls
        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])

    # End caps
    for k in range(1, n_theta - 1):
        faces.append([0, k, k + 1])
        faces.append([n_theta, n_theta + k + 1, n_theta + k])

    return vertices, np.array(faces, dtype=int)


def assemble_impeller(v_single, f_single, r_hub_stagen):
    """Replicates blades around wheel axis and attaches customized hub."""
    all_verts, all_faces = [], []
    v_len = len(v_single)

    # 1. Pattern blades radially
    for k in range(N_BLADES):
        angle = 2.0 * math.pi * k / N_BLADES
        ca, sa = math.cos(angle), math.sin(angle)

        v_rot = v_single.copy()
        v_rot[:, 1] = ca * v_single[:, 1] - sa * v_single[:, 2]
        v_rot[:, 2] = sa * v_single[:, 1] + ca * v_single[:, 2]

        all_verts.append(v_rot)
        all_faces.append(f_single + k * v_len)

    # 2. Attach hub cylinder
    if INCLUDE_HUB:
        # Determine hub radius
        hub_r = CUSTOM_HUB_RADIUS if CUSTOM_HUB_RADIUS is not None else (r_hub_stagen * 0.98)

        # Determine hub axial bounds
        x_min = v_single[:, 0].min() - HUB_AXIAL_EXTENSION_UPSTREAM
        x_max = v_single[:, 0].max() + HUB_AXIAL_EXTENSION_DOWNSTREAM

        v_hub, f_hub = build_hub_disk(hub_r, x_min, x_max)

        offset = sum(len(v) for v in all_verts)
        all_verts.append(v_hub)
        all_faces.append(f_hub + offset)

    vertices = np.vstack(all_verts) * MODEL_SCALE
    faces = np.vstack(all_faces)

    return vertices, faces


# ============================================================
# EXPORTERS
# ============================================================

def write_stl(filename, vertices, faces):
    """Export ASCII STL file."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    with open(filename, "w") as f:
        f.write("solid IMPELLER_6BLADE\n")
        for i in range(len(faces)):
            f.write(f"  facet normal {normals[i, 0]:.9e} {normals[i, 1]:.9e} {normals[i, 2]:.9e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[i, 0]:.9e} {a[i, 1]:.9e} {a[i, 2]:.9e}\n")
            f.write(f"      vertex {b[i, 0]:.9e} {b[i, 1]:.9e} {b[i, 2]:.9e}\n")
            f.write(f"      vertex {c[i, 0]:.9e} {c[i, 1]:.9e} {c[i, 2]:.9e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid IMPELLER_6BLADE\n")


def write_obj(filename, vertices, faces):
    """Export Wavefront OBJ file."""
    with open(filename, "w") as f:
        f.write("# Parametric 6-Blade Impeller Geometry\n")
        np.savetxt(f, vertices, fmt="v %.9e %.9e %.9e")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print(" PARAMETRIC STAGEN IMPELLER CONVERTER")
    print("=" * 60)

    try:
        data = parse_stagen_stream_surface(INPUT_FILE)
        v_single, f_single, r_hub = build_single_blade(data)
        vertices, faces = assemble_impeller(v_single, f_single, r_hub)

        write_stl(OUTPUT_STL, vertices, faces)
        write_obj(OUTPUT_OBJ, vertices, faces)

    except Exception as e:
        print(f"\nExecution Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully created files:")
    print(f"  - STL: {OUTPUT_STL}")
    print(f"  - OBJ: {OUTPUT_OBJ}")
    print(f"Total Vertices : {len(vertices)}")
    print(f"Total Triangles: {len(faces)}\n")


if __name__ == "__main__":
    main()