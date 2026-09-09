#!/usr/bin/env python3
"""
STAGEN STREAM SURFACE -> 6-BLADE SOLID MESH GENERATOR (BLADES ONLY)

Outputs:
    - STL: impeller_blades_only.stl
    - OBJ: impeller_blades_only.obj
"""

import math
import re
import struct
import sys
from pathlib import Path
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = "stagen.out"
OUTPUT_STL = "impeller_blades_only.stl"
OUTPUT_OBJ = "impeller_blades_only.obj"

N_BLADES = 6          # 6-blade rotor configuration
MAX_THICKNESS = 0.08  # Max profile thickness ratio (8% chord)
SPAN_HEIGHT = 0.25    # Radial length of blade span
N_SPAN = 15           # Radial layers along span
MODEL_SCALE = 1.0     # Unit scaling multiplier


# ============================================================
# PARSER & GENERATOR
# ============================================================

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_stagen_stream_surface(filename):
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

    pts = np.array(pts)
    return {"X": pts[:, 0], "Y": pts[:, 1], "R": pts[:, 2]}


def generate_closed_airfoil(x, y, max_t_ratio):
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

    # Side walls
    for s in range(N_SPAN - 1):
        for k in range(n_pts):
            k_next = (k + 1) % n_pts
            p0 = s * n_pts + k
            p1 = s * n_pts + k_next
            p2 = (s + 1) * n_pts + k_next
            p3 = (s + 1) * n_pts + k

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # End caps with correct outward normals
    for k in range(1, n_pts - 1):
        faces.append([0, k, k + 1])
        top_off = (N_SPAN - 1) * n_pts
        faces.append([top_off, top_off + k + 1, top_off + k])

    return vertices, np.array(faces, dtype=int)


def assemble_blades(v_single, f_single):
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

    return np.vstack(all_verts) * MODEL_SCALE, np.vstack(all_faces)


def write_binary_stl(filename, vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    num_triangles = len(faces)
    with open(filename, "wb") as f:
        f.write((b"STAGEN Blades Mesh " + b"\x00" * 80)[:80])
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


def write_obj(filename, vertices, faces):
    with open(filename, "w") as f:
        f.write("# Blades Mesh\n")
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def main():
    try:
        data = parse_stagen_stream_surface(INPUT_FILE)
        v_single, f_single = build_single_blade(data)
        vertices, faces = assemble_blades(v_single, f_single)

        write_binary_stl(OUTPUT_STL, vertices, faces)
        write_obj(OUTPUT_OBJ, vertices, faces)
        print(f"Generated: {OUTPUT_STL} ({Path(OUTPUT_STL).stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()