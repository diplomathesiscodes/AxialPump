#!/usr/bin/env python3
"""
STAGEN AXIAL PUMP BLADE GENERATOR
Preserves global axis of rotation and lofts clean 3D solid STL models.
"""

import math
import struct
import re
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "stagen.out"
SINGLE_BLADE_STL = "single_blade_pump.stl"
FULL_IMPELLER_STL = "impeller_pump.stl"

N_BLADES = 6            # Total blades in impeller
BLADE_THICKNESS = 0.003  # Offset thickness in meters (3 mm)
N_PROFILE_POINTS = 100  # Resampled points per side (200 total per closed section)


# ============================================================
# STAGEN PARSER (GLOBAL ROTATION AXIS PRESERVED)
# ============================================================

def parse_stagen_sections(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    content = path.read_text(errors="ignore")
    lines = content.splitlines()

    raw_tables = []
    current_table = []

    num_pattern = re.compile(
        r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    for line in lines:
        match = num_pattern.match(line)
        if match:
            v1, v2, v3 = float(match.group(1)), float(match.group(2)), float(match.group(3))
            current_table.append([v1, v2, v3])
        else:
            if len(current_table) >= 8:
                raw_tables.append(np.array(current_table))
            current_table = []

    if len(current_table) >= 8:
        raw_tables.append(np.array(current_table))

    if not raw_tables:
        raise RuntimeError(f"No coordinate tables found in '{filename}'.")

    processed = []
    for tab in raw_tables:
        c0, c1, c2 = tab[:, 0], tab[:, 1], tab[:, 2]

        # Identify Radius R (column with largest mean magnitude)
        r_idx = int(np.argmax([np.mean(np.abs(c0)), np.mean(np.abs(c1)), np.mean(np.abs(c2))]))
        r_val = np.abs(tab[:, r_idx])

        other_indices = [i for i in range(3) if i != r_idx]
        col_a, col_b = tab[:, other_indices[0]], tab[:, other_indices[1]]

        # Identify Axial X (column with larger physical span)
        if np.ptp(col_a) >= np.ptp(col_b):
            x_val, th_raw = col_a, col_b
        else:
            x_val, th_raw = col_b, col_a

        # Convert R*Theta to pure Theta in radians if required
        r_safe = np.maximum(r_val, 1e-5)
        if np.max(np.abs(th_raw)) > 2.0 * np.pi:
            theta_val = th_raw / r_safe
        else:
            theta_val = th_raw

        # Sort points continuously along chord from Leading Edge to Trailing Edge
        s_arc = np.cumsum(np.hypot(np.gradient(x_val), np.gradient(theta_val * r_val)))
        sort_idx = np.argsort(s_arc)
        x_val = x_val[sort_idx]
        theta_val = theta_val[sort_idx]
        r_val = r_val[sort_idx]

        # Uniformly resample profile points
        t_old = np.linspace(0, 1, len(x_val))
        t_new = np.linspace(0, 1, N_PROFILE_POINTS)

        x_res = np.interp(t_new, t_old, x_val)
        th_res = np.interp(t_new, t_old, theta_val)
        r_res = np.interp(t_new, t_old, r_val)

        processed.append({
            "mean_r": np.mean(r_res),
            "X": x_res,        # Keep absolute axial position
            "Theta": th_res,    # Keep absolute tangential angle
            "R": r_res         # Keep absolute radial distance from centerline
        })

    # Sort stream sections strictly Hub to Tip
    processed.sort(key=lambda s: s["mean_r"])
    return processed


# ============================================================
# SOLID AIRFOIL GENERATION & LOFTING
# ============================================================

def create_closed_profile(sec, max_t):
    x = sec["X"]
    r = sec["R"]
    s = sec["Theta"] * r  # Physical arc length

    dx = np.gradient(x)
    ds = np.gradient(s)
    length = np.hypot(dx, ds)
    length[length == 0] = 1e-12

    nx = -ds / length
    ns = dx / length

    # Smooth thickness tapering at LE and TE
    t_dist = np.linspace(0, 1, len(x))
    thickness = 2.0 * max_t * np.sin(np.pi * t_dist)

    x_upper = x + 0.5 * thickness * nx
    s_upper = s + 0.5 * thickness * ns

    x_lower = x - 0.5 * thickness * nx
    s_lower = s - 0.5 * thickness * ns

    x_closed = np.concatenate([x_upper, x_lower[::-1]])
    s_closed = np.concatenate([s_upper, s_lower[::-1]])
    r_closed = np.concatenate([r, r[::-1]])

    theta_closed = s_closed / np.maximum(r_closed, 1e-5)

    return x_closed, theta_closed, r_closed


def build_blade_mesh(sections):
    n_sec = len(sections)
    profiles = [create_closed_profile(s, BLADE_THICKNESS) for s in sections]

    n_pts_prof = len(profiles[0][0])
    grid = np.zeros((n_sec, n_pts_prof, 3))

    max_r = max(s["mean_r"] for s in sections)
    scale = 1000.0 if max_r < 5.0 else 1.0  # Scale meters to mm for standard CAD

    for s_idx, (x_p, theta_p, r_p) in enumerate(profiles):
        grid[s_idx, :, 0] = (r_p * np.cos(theta_p)) * scale  # X CAD
        grid[s_idx, :, 1] = (r_p * np.sin(theta_p)) * scale  # Y CAD
        grid[s_idx, :, 2] = x_p * scale                      # Z CAD (Axial Shaft Axis)

    vertices = grid.reshape(-1, 3)
    faces = []

    # Loft spanwise sections
    for s in range(n_sec - 1):
        for k in range(n_pts_prof):
            k_next = (k + 1) % n_pts_prof
            p0 = s * n_pts_prof + k
            p1 = s * n_pts_prof + k_next
            p2 = (s + 1) * n_pts_prof + k_next
            p3 = (s + 1) * n_pts_prof + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # End caps for Hub and Tip
    def build_cap(offset, reverse=False):
        cap_faces = []
        n_side = N_PROFILE_POINTS
        for i in range(n_side - 1):
            p_top_a = offset + i
            p_top_b = offset + i + 1
            p_bot_a = offset + (2 * n_side - 1 - i)
            p_bot_b = offset + (2 * n_side - 2 - i)

            if not reverse:
                cap_faces.append([p_top_a, p_bot_a, p_top_b])
                cap_faces.append([p_top_b, p_bot_a, p_bot_b])
            else:
                cap_faces.append([p_top_a, p_top_b, p_bot_a])
                cap_faces.append([p_top_b, p_bot_b, p_bot_a])
        return cap_faces

    faces.extend(build_cap(0, reverse=False))
    faces.extend(build_cap((n_sec - 1) * n_pts_prof, reverse=True))

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=int)


def replicate_impeller(v_single, f_single, num_blades):
    all_v, all_f = [], []
    v_len = len(v_single)

    for k in range(num_blades):
        angle = 2.0 * math.pi * k / num_blades
        ca, sa = math.cos(angle), math.sin(angle)

        v_rot = v_single.copy()
        v_rot[:, 0] = ca * v_single[:, 0] - sa * v_single[:, 1]
        v_rot[:, 1] = sa * v_single[:, 0] + ca * v_single[:, 1]

        all_v.append(v_rot)
        all_f.append(f_single + k * v_len)

    return np.vstack(all_v), np.vstack(all_f)


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
        f.write((b"STAGEN Axial Pump Solid STL" + b"\x00" * 80)[:80])
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
# MAIN
# ============================================================

def main():
    print(f"Parsing '{INPUT_FILE}'...")
    sections = parse_stagen_sections(INPUT_FILE)
    print(f" Loaded {len(sections)} stream sections from Hub to Tip.")

    print("Lofting 3D blade mesh...")
    v_single, f_single = build_blade_mesh(sections)
    write_binary_stl(SINGLE_BLADE_STL, v_single, f_single)
    print(f" Saved: {SINGLE_BLADE_STL}")

    print(f"Replicating {N_BLADES}-blade full impeller...")
    v_full, f_full = replicate_impeller(v_single, f_single, N_BLADES)
    write_binary_stl(FULL_IMPELLER_STL, v_full, f_full)
    print(f" Saved: {FULL_IMPELLER_STL}")

if __name__ == "__main__":
    main()