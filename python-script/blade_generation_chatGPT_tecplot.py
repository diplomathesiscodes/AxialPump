#!/usr/bin/env python3
"""
extract_blade.py
================

Extract candidate bare-metal blade surfaces from a cleaned
MULTALL/Tecplot XYZ file.

INPUT
-----
tecplot-input-shrinked.dat

The input file is expected to contain ONLY the X Y Z coordinate
values from the original MULTALL Tecplot zone.

Original MULTALL grid:

    I = 46
    J = 141
    K = 46

Therefore:

    46 * 141 * 46 = 298,716 points

and:

    298,716 * 3 = 896,148 numerical values

The script does NOT run STAGEN or MULTALL.

It reads the XYZ data, reconstructs the structured grid, examines
the six outer grid surfaces and produces diagnostic files that allow
us to identify the actual blade metal surfaces.

INPUT DIRECTORY EXAMPLE
-----------------------

    AxialPump/
        extract_blade.py
        tecplot-input-shrinked.dat

RUN
---

    python3 extract_blade.py

or:

    python3 extract_blade.py \
        --input-file tecplot-input-shrinked.dat \
        --output-dir blade_output

OUTPUT
------

    blade_output/
        all_points.xyz
        candidate_faces.csv
        selected_faces.csv

        I_1.obj
        I_1.stl
        I_46.obj
        I_46.stl
        ...

        blade_surface.obj
        blade_surface_ascii.stl

        diagnostics/
            candidate_I_1_meridional.png
            candidate_I_1_3d.png
            ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# MULTALL GRID SIZE
# ============================================================

NI = 46
NJ = 141
NK = 46

INPUT_FILENAME = "tecplot-input-shrinked.dat"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract candidate bare-metal blade surfaces "
            "from MULTALL XYZ data."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(INPUT_FILENAME),
        help=(
            "Cleaned MULTALL XYZ file. "
            "Default: tecplot-input-shrinked.dat"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blade_output"),
        help="Directory for generated files.",
    )

    parser.add_argument(
        "--no-save-points",
        action="store_true",
        help="Do not write all_points.xyz.",
    )

    return parser.parse_args()


# ============================================================
# READ SINGLE INPUT FILE
# ============================================================

def read_input_file(filename: Path) -> bytes:
    if not filename.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n\n"
            f"    {filename.resolve()}\n\n"
            f"Expected:\n"
            f"    {INPUT_FILENAME}\n"
        )

    if not filename.is_file():
        raise RuntimeError(
            f"Input path is not a file:\n"
            f"    {filename.resolve()}"
        )

    size_mb = filename.stat().st_size / (1024.0 * 1024.0)

    print("\nInput file")
    print("----------")
    print(f"File : {filename.resolve()}")
    print(f"Size : {size_mb:.2f} MB")

    print("\nReading file...")

    data = filename.read_bytes()

    print(
        f"Read {len(data):,} bytes."
    )

    return data


# ============================================================
# NUMBER PARSER
# ============================================================

FLOAT_RE = re.compile(
    rb"""
    [+-]?
    (?:
        (?:\d+\.\d*)
        |
        (?:\.\d+)
        |
        (?:\d+)
    )
    (?:
        [EeDd][+-]?\d+
    )?
    """,
    re.VERBOSE,
)


def extract_numbers(raw: bytes) -> np.ndarray:
    print("\nParsing numerical values...")

    matches = FLOAT_RE.findall(raw)

    found = len(matches)

    expected_points = NI * NJ * NK

    expected_values = expected_points * 3

    print(
        f"Values found    : {found:,}"
    )

    print(
        f"Values expected  : {expected_values:,}"
    )

    print(
        f"Points expected  : {expected_points:,}"
    )

    if found != expected_values:
        raise ValueError(
            "\nThe number of numerical values is incorrect.\n\n"
            f"Found    : {found:,}\n"
            f"Expected : {expected_values:,}\n\n"
            "The file must contain exactly the cleaned "
            "X Y Z coordinate data for one "
            "46 x 141 x 46 MULTALL zone.\n\n"
            "Check that the Tecplot header and any other "
            "variables have been removed."
        )

    values = np.fromiter(
        (
            float(
                token
                .replace(b"D", b"E")
                .replace(b"d", b"e")
            )
            for token in matches
        ),
        dtype=np.float64,
        count=found,
    )

    if not np.all(np.isfinite(values)):
        bad = np.count_nonzero(
            ~np.isfinite(values)
        )

        raise ValueError(
            f"\nThe input still contains {bad} "
            "non-finite numerical values.\n\n"
            "Remove NaN/Inf values before running "
            "the geometry extractor."
        )

    return values


# ============================================================
# RECONSTRUCT STRUCTURED GRID
# ============================================================

def build_grid(values: np.ndarray) -> np.ndarray:
    expected_points = NI * NJ * NK

    points = values.reshape(
        (expected_points, 3)
    )

    """
    MULTALL/Tecplot structured grid.

    We preserve the Fortran-style point ordering:

        i + NI * (j + NJ * k)

    Result:

        grid[i, j, k, coordinate]
    """

    grid = points.reshape(
        (NI, NJ, NK, 3),
        order="F",
    )

    return grid


# ============================================================
# BASIC GEOMETRY
# ============================================================

def coordinate_statistics(grid):
    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]

    radius = np.sqrt(
        x*x + y*y
    )

    print("\nCoordinate ranges")
    print("-----------------")

    print(
        f"X      : {x.min(): .12g}"
        f" ... {x.max(): .12g}"
    )

    print(
        f"Y      : {y.min(): .12g}"
        f" ... {y.max(): .12g}"
    )

    print(
        f"Z      : {z.min(): .12g}"
        f" ... {z.max(): .12g}"
    )

    print(
        f"Radius : {radius.min(): .12g}"
        f" ... {radius.max(): .12g}"
    )

    return x, y, z, radius


# ============================================================
# SIX OUTER SURFACES
# ============================================================

FACE_DEFINITIONS = {
    "I=1":
        ("I", 0),

    "I=46":
        ("I", NI - 1),

    "J=1":
        ("J", 0),

    "J=141":
        ("J", NJ - 1),

    "K=1":
        ("K", 0),

    "K=46":
        ("K", NK - 1),
}


def get_face(
    grid: np.ndarray,
    face_name: str,
) -> np.ndarray:
    direction, index = (
        FACE_DEFINITIONS[face_name]
    )

    if direction == "I":
        return grid[
            index,
            :,
            :,
            :,
        ]

    if direction == "J":
        return grid[
            :,
            index,
            :,
            :,
        ]

    if direction == "K":
        return grid[
            :,
            :,
            index,
            :,
        ]

    raise ValueError(
        f"Unknown face: {face_name}"
    )


# ============================================================
# SURFACE AREA
# ============================================================

def surface_area(surface):
    p00 = surface[:-1, :-1]

    p10 = surface[1:, :-1]

    p01 = surface[:-1, 1:]

    p11 = surface[1:, 1:]

    a = p10 - p00

    b = p01 - p00

    c = p11 - p10

    d = p11 - p01

    area1 = 0.5 * np.linalg.norm(
        np.cross(a, b),
        axis=-1,
    )

    area2 = 0.5 * np.linalg.norm(
        np.cross(c, d),
        axis=-1,
    )

    return float(
        np.nansum(
            area1 + area2
        )
    )


# ============================================================
# NORMAL VARIATION
# ============================================================

def normal_variation(surface):
    a = (
        surface[1:, :-1]
        -
        surface[:-1, :-1]
    )

    b = (
        surface[:-1, 1:]
        -
        surface[:-1, :-1]
    )

    normals = np.cross(
        a,
        b,
    )

    magnitude = np.linalg.norm(
        normals,
        axis=-1,
    )

    good = magnitude > 1.0e-14

    if not np.any(good):
        return np.inf

    normals = (
        normals[good]
        /
        magnitude[good, None]
    )

    average = np.mean(
        normals,
        axis=0,
    )

    average_magnitude = np.linalg.norm(
        average
    )

    return float(
        1.0 -
        min(
            average_magnitude,
            1.0,
        )
    )


# ============================================================
# FACE RANGES
# ============================================================

def face_radius_range(surface):
    x = surface[..., 0]

    y = surface[..., 1]

    radius = np.sqrt(
        x*x + y*y
    )

    return (
        float(np.min(radius)),
        float(np.max(radius)),
    )


def face_z_range(surface):
    z = surface[..., 2]

    return (
        float(np.min(z)),
        float(np.max(z)),
    )


# ============================================================
# INSPECT ALL SIX SURFACES
# ============================================================

def inspect_faces(grid):
    rows = []

    print("\nCandidate boundary surfaces")
    print("===========================")

    for name in FACE_DEFINITIONS:
        surface = get_face(
            grid,
            name,
        )

        area = surface_area(
            surface
        )

        variation = normal_variation(
            surface
        )

        rmin, rmax = (
            face_radius_range(
                surface
            )
        )

        zmin, zmax = (
            face_z_range(
                surface
            )
        )

        row = (
            name,
            area,
            variation,
            rmin,
            rmax,
            zmin,
            zmax,
        )

        rows.append(row)

        print(
            f"{name:7s} "
            f"area={area:12.6g} "
            f"normalVar={variation:9.6f} "
            f"R={rmin:10.6f}"
            f"..{rmax:10.6f} "
            f"Z={zmin:10.6f}"
            f"..{zmax:10.6f}"
        )

    return rows


# ============================================================
# WRITE FACE CSV
# ============================================================

def write_face_csv(
    filename,
    rows,
):
    with filename.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "face,area,normal_variation,"
            "r_min,r_max,z_min,z_max\n"
        )

        for row in rows:
            (
                name,
                area,
                variation,
                rmin,
                rmax,
                zmin,
                zmax,
            ) = row

            f.write(
                f"{name},"
                f"{area:.12e},"
                f"{variation:.12e},"
                f"{rmin:.12e},"
                f"{rmax:.12e},"
                f"{zmin:.12e},"
                f"{zmax:.12e}\n"
            )


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_surface(
    surface,
    filename,
    title,
):
    x = surface[..., 0]

    y = surface[..., 1]

    z = surface[..., 2]

    radius = np.sqrt(
        x*x + y*y
    )

    theta = np.degrees(
        np.arctan2(y, x)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].scatter(
        z.ravel(),
        radius.ravel(),
        s=2,
    )

    axes[0].set_xlabel(
        "Z"
    )

    axes[0].set_ylabel(
        "Radius"
    )

    axes[0].set_title(
        f"{title} - meridional"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].scatter(
        theta.ravel(),
        radius.ravel(),
        s=2,
    )

    axes[1].set_xlabel(
        "Theta [deg]"
    )

    axes[1].set_ylabel(
        "Radius"
    )

    axes[1].set_title(
        f"{title} - circumferential"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=180,
    )

    plt.close(fig)


def plot_surface_3d(
    surface,
    filename,
    title,
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    x = surface[..., 0]

    y = surface[..., 1]

    z = surface[..., 2]

    fig = plt.figure(
        figsize=(9, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.scatter(
        x.ravel(),
        y.ravel(),
        z.ravel(),
        s=1,
    )

    ax.set_xlabel("X")

    ax.set_ylabel("Y")

    ax.set_zlabel("Z")

    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# STRUCTURED SURFACE -> TRIANGLES
# ============================================================

def surface_to_triangles(
    surface,
):
    n1, n2, _ = (
        surface.shape
    )

    vertices = (
        surface.reshape(
            (-1, 3)
        )
        .copy()
    )

    faces = []

    def vertex_index(i, j):
        return (
            i * n2 + j
        )

    for i in range(
        n1 - 1
    ):
        for j in range(
            n2 - 1
        ):
            a = vertex_index(
                i,
                j,
            )

            b = vertex_index(
                i + 1,
                j,
            )

            c = vertex_index(
                i + 1,
                j + 1,
            )

            d = vertex_index(
                i,
                j + 1,
            )

            faces.append(
                (a, b, c)
            )

            faces.append(
                (a, c, d)
            )

    return (
        vertices,
        np.asarray(
            faces,
            dtype=np.int64,
        ),
    )


# ============================================================
# WELD DUPLICATE VERTICES
# ============================================================

def remove_duplicate_vertices(
    vertices,
    faces,
    tolerance=1.0e-10,
):
    if len(vertices) == 0:
        return (
            vertices,
            faces,
        )

    scaled = (
        vertices
        /
        tolerance
    )

    keys = np.round(
        scaled
    ).astype(
        np.int64
    )

    unique_keys, inverse = (
        np.unique(
            keys,
            axis=0,
            return_inverse=True,
        )
    )

    unique_vertices = (
        np.zeros(
            (
                len(unique_keys),
                3,
            ),
            dtype=np.float64,
        )
    )

    counts = np.bincount(
        inverse
    )

    for c in range(3):
        sums = np.bincount(
            inverse,
            weights=vertices[:, c],
        )

        unique_vertices[:, c] = (
            sums / counts
        )

    new_faces = (
        inverse[faces]
    )

    return (
        unique_vertices,
        new_faces,
    )


# ============================================================
# REMOVE DEGENERATE TRIANGLES
# ============================================================

def remove_degenerate_faces(
    vertices,
    faces,
):
    p0 = vertices[
        faces[:, 0]
    ]

    p1 = vertices[
        faces[:, 1]
    ]

    p2 = vertices[
        faces[:, 2]
    ]

    area2 = np.linalg.norm(
        np.cross(
            p1 - p0,
            p2 - p0,
        ),
        axis=1,
    )

    good = (
        area2 > 1.0e-12
    )

    return faces[good]


# ============================================================
# OBJ EXPORT
# ============================================================

def write_obj(
    filename,
    vertices,
    faces,
):
    with filename.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "# MULTALL blade surface\n"
        )

        for p in vertices:
            f.write(
                "v "
                f"{p[0]:.12e} "
                f"{p[1]:.12e} "
                f"{p[2]:.12e}\n"
            )

        for tri in faces:
            f.write(
                "f "
                f"{tri[0] + 1} "
                f"{tri[1] + 1} "
                f"{tri[2] + 1}\n"
            )


# ============================================================
# ASCII STL EXPORT
# ============================================================

def write_ascii_stl(
    filename,
    vertices,
    faces,
    name,
):
    with filename.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"solid {name}\n"
        )

        for tri in faces:
            p0 = vertices[
                tri[0]
            ]

            p1 = vertices[
                tri[1]
            ]

            p2 = vertices[
                tri[2]
            ]

            normal = np.cross(
                p1 - p0,
                p2 - p0,
            )

            length = np.linalg.norm(
                normal
            )

            if length > 1.0e-14:
                normal /= length

            else:
                normal[:] = 0.0

            f.write(
                "  facet normal "
                f"{normal[0]:.9e} "
                f"{normal[1]:.9e} "
                f"{normal[2]:.9e}\n"
            )

            f.write(
                "    outer loop\n"
            )

            for index in tri:
                p = vertices[index]

                f.write(
                    "      vertex "
                    f"{p[0]:.9e} "
                    f"{p[1]:.9e} "
                    f"{p[2]:.9e}\n"
                )

            f.write(
                "    endloop\n"
            )

            f.write(
                "  endfacet\n"
            )

        f.write(
            f"endsolid {name}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    output_dir = (
        args.output_dir
    )

    diagnostics_dir = (
        output_dir
        /
        "diagnostics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "MULTALL BARE-METAL BLADE "
        "EXTRACTION"
    )
    print("=" * 72)

    print("\nGrid dimensions")

    print(
        f"    I = {NI}"
    )

    print(
        f"    J = {NJ}"
    )

    print(
        f"    K = {NK}"
    )

    print(
        f"    Points = "
        f"{NI * NJ * NK:,}"
    )

    # --------------------------------------------------------
    # 1. READ ONE FILE
    # --------------------------------------------------------

    raw = read_input_file(
        args.input_file
    )

    # --------------------------------------------------------
    # 2. PARSE XYZ
    # --------------------------------------------------------

    values = extract_numbers(
        raw
    )

    # --------------------------------------------------------
    # 3. BUILD STRUCTURED GRID
    # --------------------------------------------------------

    print(
        "\nReconstructing "
        "structured grid..."
    )

    grid = build_grid(
        values
    )

    print(
        f"Grid shape: "
        f"{grid.shape}"
    )

    # --------------------------------------------------------
    # 4. COORDINATE STATISTICS
    # --------------------------------------------------------

    coordinate_statistics(
        grid
    )

    # --------------------------------------------------------
    # 5. OPTIONAL COMPLETE XYZ
    # --------------------------------------------------------

    if not args.no_save_points:
        xyz_file = (
            output_dir
            /
            "all_points.xyz"
        )

        print(
            f"\nWriting {xyz_file}"
        )

        np.savetxt(
            xyz_file,
            grid.reshape(
                (-1, 3)
            ),
            fmt="%.12e",
            header="X Y Z",
            comments="",
        )

    # --------------------------------------------------------
    # 6. INSPECT SIX GRID BOUNDARIES
    # --------------------------------------------------------

    rows = inspect_faces(
        grid
    )

    write_face_csv(
        output_dir
        /
        "candidate_faces.csv",
        rows,
    )

    # --------------------------------------------------------
    # 7. PLOT ALL SIX
    # --------------------------------------------------------

    print(
        "\nCreating diagnostics..."
    )

    for face_name in (
        FACE_DEFINITIONS
    ):
        surface = get_face(
            grid,
            face_name,
        )

        safe_name = (
            face_name
            .replace(
                "=",
                "_",
            )
        )

        plot_surface(
            surface,
            diagnostics_dir
            /
            f"candidate_"
            f"{safe_name}"
            f"_meridional.png",
            face_name,
        )

        plot_surface_3d(
            surface,
            diagnostics_dir
            /
            f"candidate_"
            f"{safe_name}"
            f"_3d.png",
            face_name,
        )

    # --------------------------------------------------------
    # 8. PRELIMINARY RANKING
    # --------------------------------------------------------

    print(
        "\nPreliminary "
        "surface ranking"
    )

    print(
        "-------------------------"
    )

    normal_variations = np.array(
        [
            row[2]
            for row in rows
        ]
    )

    minimum = (
        normal_variations.min()
    )

    maximum = (
        normal_variations.max()
    )

    if (
        maximum - minimum
        >
        1.0e-14
    ):
        normalized = (
            normal_variations
            -
            minimum
        ) / (
            maximum
            -
            minimum
        )

    else:
        normalized = np.zeros(
            len(rows)
        )

    ranking = []

    for row, score in zip(
        rows,
        normalized,
    ):
        ranking.append(
            (
                row[0],
                float(score),
            )
        )

    ranking.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for n, (
        name,
        score,
    ) in enumerate(
        ranking,
        1,
    ):
        print(
            f"{n}. "
            f"{name:7s} "
            f"score={score:.5f}"
        )

    # --------------------------------------------------------
    # 9. SAVE RANKING
    # --------------------------------------------------------

    selected_file = (
        output_dir
        /
        "selected_faces.csv"
    )

    with selected_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "rank,face,score\n"
        )

        for n, (
            name,
            score,
        ) in enumerate(
            ranking,
            1,
        ):
            f.write(
                f"{n},"
                f"{name},"
                f"{score:.12e}\n"
            )

    # --------------------------------------------------------
    # 10. CREATE MESH FOR ALL SIX CANDIDATES
    #
    # We save ALL six here rather than automatically selecting
    # two potentially incorrect surfaces.
    # --------------------------------------------------------

    print(
        "\nWriting individual "
        "candidate surfaces..."
    )

    for face_name in (
        FACE_DEFINITIONS
    ):
        surface = get_face(
            grid,
            face_name,
        )

        vertices, faces = (
            surface_to_triangles(
                surface
            )
        )

        (
            vertices,
            faces,
        ) = remove_duplicate_vertices(
            vertices,
            faces,
        )

        faces = (
            remove_degenerate_faces(
                vertices,
                faces,
            )
        )

        safe_name = (
            face_name
            .replace(
                "=",
                "_",
            )
        )

        write_obj(
            output_dir
            /
            f"{safe_name}.obj",
            vertices,
            faces,
        )

        write_ascii_stl(
            output_dir
            /
            f"{safe_name}.stl",
            vertices,
            faces,
            name=safe_name,
        )

    # --------------------------------------------------------
    # 11. FINISH
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nInput read:"
    )

    print(
        f"    {args.input_file.resolve()}"
    )

    print(
        "\nNo other input data files were read."
    )

    print(
        "\nNo STAGEN/MULTALL executable was run."
    )

    print(
        "\nDiagnostic output:"
    )

    print(
        f"    {diagnostics_dir.resolve()}"
    )

    print(
        "\nCandidate surface meshes:"
    )

    for face_name in (
        FACE_DEFINITIONS
    ):
        safe_name = (
            face_name
            .replace(
                "=",
                "_",
            )
        )

        print(
            f"    {safe_name}.obj"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The six candidate surfaces are "
        "NOT automatically assumed to be "
        "blade surfaces."
    )

    print(
        "Inspect the diagnostic plots before "
        "using a candidate as the bare-metal blade."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user."
        )

        sys.exit(130)

    except Exception as exc:
        print(
            "\nERROR:"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "\nThe script did not execute "
            "STAGEN or MULTALL."
        )

        sys.exit(1)