#!/usr/bin/env python3
"""
STAGEN STREAM SURFACE -> SOLID HUB CYLINDER GENERATOR (HUB ONLY)

Outputs:
    - STL: impeller_hub_only.stl
    - OBJ: impeller_hub_only.obj
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
OUTPUT_STL = "impeller_hub_only_scale.stl"
OUTPUT_OBJ = "impeller_hub_only_scale.obj"

# --- Shaft & Hub Sizing ---
CUSTOM_HUB_RADIUS = 0.50    #None  # Float value (e.g., 0.05) or None to auto-scale from STAGEN
HUB_AXIAL_EXTENSION_UPSTREAM = 0.01
HUB_AXIAL_EXTENSION_DOWNSTREAM = 0.01

N_THETA_HUB = 64      # Radial resolution of hub cylinder
MODEL_SCALE = 1.0     # Unit scaling multiplier


# ============================================================
# PARSER & GENERATOR
# ============================================================

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_stagen_bounds(filename):
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
    return {
        "x_min": pts[:, 0].min(),
        "x_max": pts[:, 0].max(),
        "r_hub": float(np.mean(pts[:, 2]))
    }


def build_hub_disk(radius, x_min, x_max, n_theta=N_THETA_HUB):
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

    vertices = np.vstack([v_hub_bot, v_hub_top]) * MODEL_SCALE
    faces = []

    # Cylinder side walls
    for k in range(n_theta):
        k_next = (k + 1) % n_theta
        b0, b1 = k, k_next
        t0, t1 = n_theta + k, n_theta + k_next

        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])

    # End caps with correct outward normals
    for k in range(1, n_theta - 1):
        faces.append([0, k + 1, k])
        faces.append([n_theta, n_theta + k, n_theta + k + 1])

    return vertices, np.array(faces, dtype=int)


def write_binary_stl(filename, vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals /= norms

    num_triangles = len(faces)
    with open(filename, "wb") as f:
        f.write((b"STAGEN Hub Mesh " + b"\x00" * 80)[:80])
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
        f.write("# Hub Mesh\n")
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def main():
    try:
        bounds = parse_stagen_bounds(INPUT_FILE)
        hub_r = CUSTOM_HUB_RADIUS if CUSTOM_HUB_RADIUS is not None else (bounds["r_hub"] * 0.98)
        x_min = bounds["x_min"] - HUB_AXIAL_EXTENSION_UPSTREAM
        x_max = bounds["x_max"] + HUB_AXIAL_EXTENSION_DOWNSTREAM

        vertices, faces = build_hub_disk(hub_r, x_min, x_max)

        write_binary_stl(OUTPUT_STL, vertices, faces)
        write_obj(OUTPUT_OBJ, vertices, faces)
        print(f"Generated: {OUTPUT_STL} ({Path(OUTPUT_STL).stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
