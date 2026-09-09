#!/usr/bin/env python3
"""
STAGEN STREAM SURFACE -> COMPACT BINARY 6-BLADE IMPELLER GENERATOR

Converts STAGEN stream surface data into an optimized, lightweight Binary STL
mesh suitable for CAD and CFD tools.

Outputs:
    - Compact Binary STL (.stl)
    - Wavefront OBJ (.obj)
"""

import math
import re
import struct
import sys
from pathlib import Path
import numpy as np

# ============================================================
# USER SETTINGS (IMPELLER & MESH COMPRESSION)
# ============================================================

INPUT_FILE = "stagen.out"
OUTPUT_STL = "impeller_6blade_shrink.stl"
OUTPUT_OBJ = "impeller_6blade_shrink.obj"

# --- Blade Geometry ---
N_BLADES = 6          # Number of blades around wheel (60° spacing)
MAX_THICKNESS = 0.08  # Max profile thickness ratio (8% chord)
SPAN_HEIGHT = 0.25    # Radial length of blade span

# --- Hub Cylinder Settings ---
INCLUDE_HUB = True    # Include center shaft hub
CUSTOM_HUB_RADIUS = None  # Custom radius (e.g., 0.05) or None for auto-size
HUB_AXIAL_EXTENSION_UPSTREAM = 0.01
HUB_AXIAL_EXTENSION_DOWNSTREAM = 0.01

# --- Resolution & File Size Controls ---
N_SPAN = 12           # Reduced spanwise layers (Default 12 gives clean low-MB mesh)
PROFILE_DECIMATION = 1 # Keep 1 for full points, 2 to skip every other profile point
N_THETA_HUB = 32      # Hub cylinder circumferential resolution
MODEL_SCALE = 1.0     # Unit scaling multiplier


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
    
    # Decimate input points if reduced profile density is requested
    if PROFILE_DECIMATION > 1:
        pts = pts[::PROFILE_DECIMATION]

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


def build_hub_disk(radius, x_min, x_max, n_theta=N_THETA_HUB):
    """Generates central shaft hub cylinder."""
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

        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])

    for k in range(1, n_theta - 1):
        faces.append([0, k, k + 1])
        faces.append([n_theta, n_theta + k + 1, n_theta + k])

    return vertices, np.array(faces, dtype=int)


def assemble_impeller(v_single, f_single, r_hub_stagen):
    """Replicates blades around wheel axis and attaches customized hub."""
    all_verts, all_faces = [], []
    v_len = len(v_single)

    for k in range(N_BLADES):
        angle = 2.0 * math.pi * k / N_BLADES
        ca, sa = math.cos(angle), math.sin(angle)

        v_rot = v_single.copy()
        v_rot[:, 1] = ca * v_single[:, 1] - sa * v_single[:, 2]
        v_rot[:, 2] = sa * v_single[:, 1] + ca * v_single[:, 2]

        all_verts.append(v_rot)
        all_faces.append(f_single + k * v_len)

    if INCLUDE_HUB:
        hub_r = CUSTOM_HUB_RADIUS if CUSTOM_HUB_RADIUS is not None else (r_hub_stagen * 0.98)
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
# BINARY EXPORTERS (HIGH COMPRESSION)
# ============================================================

def write_binary_stl(filename, vertices, faces):
    """Writes binary STL format to drastically reduce file size."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    num_triangles = len(faces)

    with open(filename, "wb") as f:
        # 80-byte header
        header = b"STAGEN Compact Binary STL " + b"\x00" * 54
        f.write(header[:80])

        # Triangle count (uint32)
        f.write(struct.pack("<I", num_triangles))

        # Pack triangle data: normal (3 float32), verts (9 float32), attr byte count (uint16)
        data = np.zeros((num_triangles, 12), dtype=np.float32)
        data[:, 0:3] = normals
        data[:, 3:6] = a
        data[:, 6:9] = b
        data[:, 9:12] = c

        # Convert to raw bytes with 2-byte attribute padding per triangle
        raw_bytes = bytearray()
        for i in range(num_triangles):
            raw_bytes.extend(data[i].tobytes())
            raw_bytes.extend(b"\x00\x00")

        f.write(raw_bytes)


def write_obj(filename, vertices, faces):
    """Export Wavefront OBJ file."""
    with open(filename, "w") as f:
        f.write("# Compact Impeller Geometry\n")
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" STAGEN COMPACT BINARY IMPELLER GENERATOR")
    print("=" * 60)

    try:
        data = parse_stagen_stream_surface(INPUT_FILE)
        v_single, f_single, r_hub = build_single_blade(data)
        vertices, faces = assemble_impeller(v_single, f_single, r_hub)

        write_binary_stl(OUTPUT_STL, vertices, faces)
        write_obj(OUTPUT_OBJ, vertices, faces)

    except Exception as e:
        print(f"\nExecution Error: {e}", file=sys.stderr)
        sys.exit(1)

    file_size_mb = Path(OUTPUT_STL).stat().st_size / (1024 * 1024)

    print(f"\nSuccessfully generated optimized files:")
    print(f"  - STL File: {OUTPUT_STL} ({file_size_mb:.2f} MB)")
    print(f"  - OBJ File: {OUTPUT_OBJ}")
    print(f"Total Vertices : {len(vertices)}")
    print(f"Total Triangles: {len(faces)}\n")


if __name__ == "__main__":
    main()
