#!/usr/bin/env python3
"""
STAGEN SINGLE BLADE SOLID MESH GENERATOR
- Builds 1 clean, watertight solid blade profile
- Enforces outward face normals to eliminate black shading artifacts
- Outputs single blade and full 6-blade array
"""

import math
import re
import struct
import sys
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "stagen.out"
SINGLE_BLADE_STL = "single_blade.stl"
FULL_IMPELLER_STL = "impeller_6_blades.stl"

N_BLADES = 6          # Number of blades in full array
MAX_THICKNESS = 0.08  # Max profile thickness (8% chord)
SPAN_HEIGHT = 0.25    # Radial length of blade span
N_SPAN = 20           # Radial stations along span


# ============================================================
# STAGEN PARSER & GEOMETRY
# ============================================================

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_stagen_data(filename):
    """Parse axial, tangential, and radial coordinates from stagen.out."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    lines = path.read_text(errors="ignore").splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        u = line.upper()
        if "AXIAL" in u and ("TANGENTIAL" in u or "RADIAL" in u) and "STREAM" in u:
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError("Could not find stream surface data in file.")

    row_pattern = re.compile(
        rf"^\s*(\d+)\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})"
    )

    pts = []
    for line in lines[start_idx + 1:]:
        m = row_pattern.match(line)
        if m:
            pts.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        elif pts and not line.strip():
            break

    pts = np.array(pts)
    return {"X": pts[:, 0], "Y": pts[:, 1], "R": pts[:, 2]}


def generate_closed_profile(x, y, max_t_ratio):
    """Generates a non-intersecting closed 2D airfoil loop."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12

    s = np.cumsum(ds)
    s = (s - s[0]) / s[-1]

    nx = -dy / ds
    ny = dx / ds
    norm = np.hypot(nx, ny)
    nx /= norm
    ny /= norm

    chord = max(x.max() - x.min(), 1.0)
    t_max = chord * max_t_ratio
    thick = 2.0 * t_max * np.sin(np.pi * (s ** 0.75)) * (1.0 - 0.2 * s)

    x_suction = x + 0.5 * thick * nx
    y_suction = y + 0.5 * thick * ny

    x_pressure = x - 0.5 * thick * nx
    y_pressure = y - 0.5 * thick * ny

    x_closed = np.concatenate([x_suction, x_pressure[-2:0:-1]])
    y_closed = np.concatenate([y_suction, y_pressure[-2:0:-1]])

    return x_closed, y_closed


def triangulate_cap(x, y):
    """Ear clipping triangulation for planar cap end-faces."""
    n = len(x)
    pts = np.column_stack((x, y))
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    indices = list(range(n))
    if area < 0:
        indices = indices[::-1]

    def is_convex(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0

    def point_in_tri(p, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    triangles = []
    count = 0
    while len(indices) > 3 and count < 2000:
        count += 1
        ear_found = False
        num_idx = len(indices)
        for i in range(num_idx):
            i_prev, i_curr, i_next = indices[i-1], indices[i], indices[(i+1) % num_idx]
            p_prev, p_curr, p_next = pts[i_prev], pts[i_curr], pts[i_next]

            if not is_convex(p_prev, p_curr, p_next):
                continue

            has_interior = False
            for j in range(num_idx):
                test_idx = indices[j]
                if test_idx not in (i_prev, i_curr, i_next):
                    if point_in_tri(pts[test_idx], p_prev, p_curr, p_next):
                        has_interior = True
                        break

            if not has_interior:
                triangles.append([i_prev, i_curr, i_next])
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])

    return np.array(triangles)


def enforce_outward_normals(verts, faces):
    """Flips face winding where normal points inward towards mesh center."""
    c = verts.mean(axis=0)
    v_c = verts - c

    a = v_c[faces[:, 0]]
    b = v_c[faces[:, 1]]
    c_p = v_c[faces[:, 2]]

    normals = np.cross(b - a, c_p - a)
    centers = (a + b + c_p) / 3.0
    dots = np.einsum("ij,ij->i", normals, centers)

    fixed_faces = faces.copy()
    inverted = dots < 0
    fixed_faces[inverted] = fixed_faces[inverted][:, [0, 2, 1]]
    return fixed_faces


# ============================================================
# SOLID MESH BUILDER
# ============================================================

def build_single_blade_solid(data):
    """Constructs 1 watertight, manifold solid blade."""
    x_cam, y_cam, r_base = data["X"], data["Y"], data["R"]
    r_hub = float(np.mean(r_base))

    x_2d, y_2d = generate_closed_profile(x_cam, y_cam, MAX_THICKNESS)
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

    # Side walls (suction and pressure surfaces)
    for s in range(N_SPAN - 1):
        for k in range(n_pts):
            k_next = (k + 1) % n_pts
            p0 = s * n_pts + k
            p1 = s * n_pts + k_next
            p2 = (s + 1) * n_pts + k_next
            p3 = (s + 1) * n_pts + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # End caps (hub and tip)
    cap_tris = triangulate_cap(x_2d, y_2d)
    for tri in cap_tris:
        faces.append([tri[0], tri[2], tri[1]])

    top_offset = (N_SPAN - 1) * n_pts
    for tri in cap_tris:
        faces.append([top_offset + tri[0], top_offset + tri[1], top_offset + tri[2]])

    faces = np.array(faces, dtype=int)
    faces = enforce_outward_normals(vertices, faces)

    return vertices, faces


def replicate_blade_array(v_single, f_single, num_blades):
    """Replicates 1 blade around the Z-axis array."""
    all_verts, all_faces = [], []
    v_len = len(v_single)

    for k in range(num_blades):
        angle = 2.0 * math.pi * k / num_blades
        ca, sa = math.cos(angle), math.sin(angle)

        v_rot = v_single.copy()
        v_rot[:, 1] = ca * v_single[:, 1] - sa * v_single[:, 2]
        v_rot[:, 2] = sa * v_single[:, 1] + ca * v_single[:, 2]

        all_verts.append(v_rot)
        all_faces.append(f_single + k * v_len)

    return np.vstack(all_verts), np.vstack(all_faces)


def write_binary_stl(filename, vertices, faces):
    """Export binary STL file."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    num_triangles = len(faces)
    with open(filename, "wb") as f:
        f.write((b"STAGEN Single Blade Solid " + b"\x00" * 80)[:80])
        f.write(struct.pack("<I", num_triangles))

        data = np.zeros((num_triangles, 12), dtype=np.float32)
        data[:, 0:3] = normals
        data[:, 3:6] = a
        data[:, 6:9] = b
        data[:, 9:12] = c

        raw_bytes = bytearray()
        for i in range(num_triangles):
            raw_bytes.extend(data[i].tobytes())
            raw_bytes.extend(b"\x00\x00")
        f.write(raw_bytes)


# ============================================================
# EXECUTION
# ============================================================

def main():
    print("Building 1 solid blade profile...")
    data = parse_stagen_data(INPUT_FILE)
    v_single, f_single = build_single_blade_solid(data)
    write_binary_stl(SINGLE_BLADE_STL, v_single, f_single)
    print(f" Saved: {SINGLE_BLADE_STL} ({len(v_single)} vertices, {len(f_single)} facets)")

    print("\nReplicating solid blade x 6...")
    v_array, f_array = replicate_blade_array(v_single, f_single, N_BLADES)
    write_binary_stl(FULL_IMPELLER_STL, v_array, f_array)
    print(f" Saved: {FULL_IMPELLER_STL} ({len(v_array)} vertices, {len(f_array)} facets)")

if __name__ == "__main__":
    main()