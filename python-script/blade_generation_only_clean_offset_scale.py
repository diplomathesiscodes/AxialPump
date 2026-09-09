#!/usr/bin/env python3
"""
STAGEN SOLID UNIFORM OFFSET BLADE GENERATOR
- Offsets meanline by a constant thickness (t_blade)
- Straight flat caps at leading/trailing edges
- Watertight topology for CAD/STL export
- Supports global MODEL_SCALE multiplier
"""

import math
import re
import struct
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "stagen.out"
SINGLE_BLADE_STL = "single_blade_offset_scale.stl"
FULL_IMPELLER_STL = "impeller_6_blades_offset_scale.stl"

MODEL_SCALE = 13.0       # Global multiplier for 3D scale
N_BLADES = 6            # Number of blades in full array
BLADE_THICKNESS = 0.05  # Uniform thickness offset
SPAN_HEIGHT = 0.25      # Radial height of blade span
N_SPAN = 20             # Radial stations along span


# ============================================================
# DATA PARSER
# ============================================================

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_stagen_data(filename):
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


# ============================================================
# UNIFORM OFFSET PROFILE GENERATOR
# ============================================================

def generate_offset_profile(x_mean, y_mean, thickness=0.05):
    """Generates a closed, constant-thickness profile around the meanline."""
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12

    nx = -dy / ds
    ny = dx / ds
    norm = np.hypot(nx, ny)
    nx /= norm
    ny /= norm

    half_t = thickness / 2.0

    # Side 1 (Pressure side offset)
    x_side1 = x_mean + half_t * nx
    y_side1 = y_mean + half_t * ny

    # Side 2 (Suction side offset)
    x_side2 = x_mean - half_t * nx
    y_side2 = y_mean - half_t * ny

    # Loop around: LE -> TE on side1, then TE -> LE on side2
    x_closed = np.concatenate([x_side1, x_side2[::-1]])
    y_closed = np.concatenate([y_side1, y_side2[::-1]])

    return x_closed, y_closed


# ============================================================
# TOPOLOGICAL SOLID MESH BUILDER
# ============================================================

def build_single_offset_blade(data):
    x_cam, y_cam, r_base = data["X"], data["Y"], data["R"]
    r_hub = float(np.mean(r_base))

    x_2d, y_2d = generate_offset_profile(x_cam, y_cam, thickness=BLADE_THICKNESS)
    n_pts_side = len(x_cam)
    n_total_pts = len(x_2d)

    r_stations = np.linspace(r_hub, r_hub + SPAN_HEIGHT, N_SPAN)
    grid = np.zeros((N_SPAN, n_total_pts, 3))

    for s_idx, r_val in enumerate(r_stations):
        theta = y_2d / r_hub
        grid[s_idx, :, 0] = x_2d
        grid[s_idx, :, 1] = r_val * np.cos(theta)
        grid[s_idx, :, 2] = r_val * np.sin(theta)

    vertices = grid.reshape(-1, 3)
    faces = []

    # --- Side Walls ---
    for s in range(N_SPAN - 1):
        for k in range(n_total_pts):
            k_next = (k + 1) % n_total_pts
            p0 = s * n_total_pts + k
            p1 = s * n_total_pts + k_next
            p2 = (s + 1) * n_total_pts + k_next
            p3 = (s + 1) * n_total_pts + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # --- Structured Strip Cap Triangulation ---
    def build_cap_faces(offset, is_top=False):
        cap_faces = []
        for i in range(n_pts_side - 1):
            p_top_a = offset + i
            p_top_b = offset + i + 1
            p_bot_a = offset + (2 * n_pts_side - 1 - i)
            p_bot_b = offset + (2 * n_pts_side - 2 - i)

            if not is_top:
                cap_faces.append([p_top_a, p_bot_a, p_top_b])
                cap_faces.append([p_top_b, p_bot_a, p_bot_b])
            else:
                cap_faces.append([p_top_a, p_top_b, p_bot_a])
                cap_faces.append([p_top_b, p_bot_b, p_bot_a])
        return cap_faces

    # Bottom Cap (Hub, s=0)
    faces.extend(build_cap_faces(0, is_top=False))

    # Top Cap (Tip, s=N_SPAN-1)
    top_offset = (N_SPAN - 1) * n_total_pts
    faces.extend(build_cap_faces(top_offset, is_top=True))

    # Apply global model scaling
    vertices = np.array(vertices, dtype=np.float64) * MODEL_SCALE
    faces = np.array(faces, dtype=int)

    return vertices, faces


def replicate_blade_array(v_single, f_single, num_blades):
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
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    num_triangles = len(faces)
    with open(filename, "wb") as f:
        f.write((b"STAGEN Uniform Offset Blade " + b"\x00" * 80)[:80])
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
# MAIN EXECUTION
# ============================================================

def main():
    print(f"Building uniform offset blade (MODEL_SCALE = {MODEL_SCALE})...")
    data = parse_stagen_data(INPUT_FILE)
    v_single, f_single = build_single_offset_blade(data)
    write_binary_stl(SINGLE_BLADE_STL, v_single, f_single)
    print(f" Saved: {SINGLE_BLADE_STL} ({len(v_single)} vertices, {len(f_single)} facets)")

    print("\nReplicating blade x 6 array...")
    v_array, f_array = replicate_blade_array(v_single, f_single, N_BLADES)
    write_binary_stl(FULL_IMPELLER_STL, v_array, f_array)
    print(f" Saved: {FULL_IMPELLER_STL} ({len(v_array)} vertices, {len(f_array)} facets)")

if __name__ == "__main__":
    main()
