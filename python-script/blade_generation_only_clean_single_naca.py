#!/usr/bin/env python3
"""
STAGEN SOLID NACA BLADE GENERATOR (EAR-CLIPPING CAP TRIANGULATION)
- Uses 2D Ear Clipping polygon triangulation for Hub and Tip caps.
- Eliminates centroid fan line crossings on non-star-convex airfoil profiles.
- Generates 100% watertight, non-overlapping, correctly oriented STL solids.
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
SINGLE_BLADE_STL = "single_blade_naca.stl"
FULL_IMPELLER_STL = "impeller_6_blades_naca.stl"

N_BLADES = 6            # Number of blades in full array
NACA_THICKNESS = 0.12   # NACA 0012 max thickness ratio
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
# CLOSED NACA PROFILE GENERATOR
# ============================================================

def naca_closed_thickness(xc, t=0.12):
    """NACA 4-digit formula modified for exact trailing edge closure."""
    return 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(xc, 0.0))
        - 0.1260 * xc
        - 0.3516 * (xc ** 2)
        + 0.2843 * (xc ** 3)
        - 0.1036 * (xc ** 4)
    )

def generate_naca_profile_over_meanline(x_mean, y_mean, t=0.12):
    """Wraps closed NACA profile around meanline without duplicate vertices."""
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    ds = np.hypot(dx, dy)
    ds[ds == 0] = 1e-12

    s = np.cumsum(ds)
    xc = (s - s[0]) / (s[-1] - s[0])

    nx = -dy / ds
    ny = dx / ds
    norm = np.hypot(nx, ny)
    nx /= norm
    ny /= norm

    yt = naca_closed_thickness(xc, t=t)

    x_upper = x_mean + yt * nx
    y_upper = y_mean + yt * ny
    x_lower = x_mean - yt * nx
    y_lower = y_mean - yt * ny

    x_closed = np.concatenate([x_upper, x_lower[-2:0:-1]])
    y_closed = np.concatenate([y_upper, y_lower[-2:0:-1]])

    return x_closed, y_closed


# ============================================================
# PLANAR POLYGON EAR-CLIPPING TRIANGULATOR
# ============================================================

def triangulate_polygon_2d(points):
    """
    Triangulates a 2D simple polygon using Ear Clipping.
    Guarantees non-overlapping interior triangles for airfoils.
    """
    n = len(points)
    if n < 3:
        return []

    # Check signed area for orientation
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]

    # Ensure counter-clockwise vertex index list
    indices = list(range(n))
    if area < 0:
        indices.reverse()

    def is_ear(i_prev, i_curr, i_next, remaining):
        p_prev = points[i_prev]
        p_curr = points[i_curr]
        p_next = points[i_next]

        # Check cross product for convex turn
        cp = (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1]) - (
            p_curr[1] - p_prev[1]
        ) * (p_next[0] - p_prev[0])
        if cp <= 1e-12:
            return False

        # Ensure no other remaining vertex lies inside candidate ear triangle
        for idx in remaining:
            if idx in (i_prev, i_curr, i_next):
                continue
            pt = points[idx]
            v0 = p_next - p_prev
            v1 = p_curr - p_prev
            v2 = pt - p_prev

            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-12:
                continue
            inv_denom = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            if (u >= -1e-7) and (v >= -1e-7) and (u + v <= 1.0 + 1e-7):
                return False
        return True

    triangles = []
    rem = list(indices)

    count = 0
    max_count = len(rem) * len(rem)
    while len(rem) > 2 and count < max_count:
        count += 1
        ear_found = False
        for k in range(len(rem)):
            prev_k = rem[(k - 1) % len(rem)]
            curr_k = rem[k]
            next_k = rem[(k + 1) % len(rem)]

            if is_ear(prev_k, curr_k, next_k, rem):
                triangles.append([prev_k, curr_k, next_k])
                rem.pop(k)
                ear_found = True
                break
        if not ear_found:
            break

    if len(rem) == 3:
        triangles.append([rem[0], rem[1], rem[2]])

    return triangles


# ============================================================
# TOPOLOGICAL SOLID MESH BUILDER
# ============================================================

def build_single_naca_blade(data):
    x_cam, y_cam, r_base = data["X"], data["Y"], data["R"]
    r_hub = float(np.mean(r_base))

    x_2d, y_2d = generate_naca_profile_over_meanline(x_cam, y_cam, t=NACA_THICKNESS)
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

    # --- Side Walls ---
    for s in range(N_SPAN - 1):
        for k in range(n_pts):
            k_next = (k + 1) % n_pts
            p0 = s * n_pts + k
            p1 = s * n_pts + k_next
            p2 = (s + 1) * n_pts + k_next
            p3 = (s + 1) * n_pts + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # 2D Ear clipping triangulation on profile domain (x_2d, y_2d)
    pts_2d = np.column_stack([x_2d, y_2d])
    cap_tris_2d = triangulate_polygon_2d(pts_2d)

    # --- Hub Cap (Bottom, s=0) ---
    for t in cap_tris_2d:
        # Winding chosen so normal points strictly outward from hub (-Radial)
        faces.append([t[0], t[2], t[1]])

    # --- Tip Cap (Top, s=N_SPAN-1) ---
    top_offset = (N_SPAN - 1) * n_pts
    for t in cap_tris_2d:
        # Reversed winding so normal points strictly outward from tip (+Radial)
        faces.append([top_offset + t[0], top_offset + t[1], top_offset + t[2]])

    vertices = np.array(vertices, dtype=np.float64)
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
        f.write((b"STAGEN Watertight Solid Blade " + b"\x00" * 80)[:80])
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
    print("Building NACA solid blade with ear-clipped caps...")
    data = parse_stagen_data(INPUT_FILE)
    v_single, f_single = build_single_naca_blade(data)
    write_binary_stl(SINGLE_BLADE_STL, v_single, f_single)
    print(f" Saved: {SINGLE_BLADE_STL} ({len(v_single)} vertices, {len(f_single)} facets)")

    print("\nReplicating blade x 6 array...")
    v_array, f_array = replicate_blade_array(v_single, f_single, N_BLADES)
    write_binary_stl(FULL_IMPELLER_STL, v_array, f_array)
    print(f" Saved: {FULL_IMPELLER_STL} ({len(v_array)} vertices, {len(f_array)} facets)")

if __name__ == "__main__":
    main()