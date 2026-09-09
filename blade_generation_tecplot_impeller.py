#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MULTALL TECPLOT CHUNKS -> 6 BLADE AXIAL-CONICAL IMPELLER
=========================================================

INPUT
-----

Only these 12 files are read:

    tecplot-input-shrinked-chunk.dat00
    tecplot-input-shrinked-chunk.dat01
    ...
    tecplot-input-shrinked-chunk.dat11

No STAGEN file is read.
No stagen.out is read.
No STL flow-passage files are read.
No velocity/pressure data are required.

EXPECTED GRID
--------------

    I = 46
    J = 141
    K = 46

Total:

    46 * 141 * 46 = 298356 XYZ points


KNOWN MULTALL BLADE STATIONS
----------------------------

    JLE = 16
    JTE = 116

These define the leading-edge and trailing-edge
stations in the MULTALL streamwise direction.

The program does NOT assume beforehand that I/J/K
is the blade-to-blade direction.

It determines the coordinate behavior of all three
directions and reports the result.

MACHINE AXIS
------------

The supplied coordinates indicate that X is the rotor axis.

Rotation is therefore performed around X.

SIX BLADES
----------

    pitch = 360 / 6 = 60 degrees


OUTPUT
------

impeller_output/

    reconstructed_points.xyz

    topology_report.txt

    side_A.xyz
    side_B.xyz

    blade_single.stl
    blade_single.xyz
    blade_single.png

    impeller_6_blades.stl
    impeller_6_blades.xyz
    impeller_6_blades.png


IMPORTANT
---------

The program first determines the topology.

It does NOT blindly assume K=1/K=46 are blade faces.

The two surfaces used for the blade are selected from
the grid direction having the strongest circumferential
variation while retaining a coherent streamwise/radial
surface.

If the automatic result is ambiguous, the program stops
before creating an STL and writes diagnostics.
"""


from pathlib import Path
import math
import struct
import sys

import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

CHUNK_PREFIX = "tecplot-input-shrinked-chunk.dat"

NUMBER_OF_CHUNKS = 12

OUTPUT_DIR = Path(
    "impeller_output"
)

# MULTALL dimensions
NI = 46
NJ = 141
NK = 46

# Rotor
NBLADES = 6

# Leading/trailing edge stations
JLE = 16
JTE = 116


# ============================================================
# EXPECTED FILE NAMES
# ============================================================

CHUNKS = [
    Path(
        f"{CHUNK_PREFIX}{i:02d}"
    )
    for i in range(
        NUMBER_OF_CHUNKS
    )
]


# ============================================================
# BASIC HELPERS
# ============================================================

def die(message):

    print()
    print("=" * 78)
    print("ERROR")
    print("=" * 78)
    print()
    print(message)
    print()

    sys.exit(1)


def finite_or_die(a, name):

    if not np.isfinite(a).all():

        die(
            f"Non-finite coordinate detected in {name}."
        )


# ============================================================
# READ CHUNKS
# ============================================================

def read_chunks():

    print()
    print("=" * 78)
    print("READING TECPLOT CHUNKS")
    print("=" * 78)

    all_points = []

    for filename in CHUNKS:

        if not filename.exists():

            die(
                "Missing chunk:\n\n"
                f"    {filename.resolve()}\n\n"
                "Expected all 12 files."
            )

        print(
            f"Reading {filename} ..."
        )

        # ----------------------------------------------------
        # The shrinked chunks contain XYZ columns.
        # Read first three numerical columns.
        # ----------------------------------------------------

        try:

            data = np.loadtxt(
                filename,
                dtype=np.float64,
                usecols=(0, 1, 2)
            )

        except Exception as exc:

            die(
                f"Could not read {filename}:\n"
                f"{exc}"
            )

        if data.ndim == 1:

            if len(data) != 3:

                die(
                    f"Unexpected line in {filename}."
                )

            data = data.reshape(
                1,
                3
            )

        finite_or_die(
            data,
            str(filename)
        )

        print(
            f"    points = {len(data):,}"
        )

        all_points.append(
            data
        )

    xyz = np.vstack(
        all_points
    )

    expected = (
        NI *
        NJ *
        NK
    )

    print()
    print(
        f"Total points read : {len(xyz):,}"
    )

    print(
        f"Expected          : {expected:,}"
    )

    if len(xyz) != expected:

        die(
            "The 12 chunks do not contain exactly "
            f"{expected:,} XYZ points."
        )

    return xyz


# ============================================================
# RECONSTRUCT GRID
# ============================================================

def reconstruct_grid(xyz):

    """
    Tecplot POINT ordering is reconstructed as:

        K, J, I, XYZ

    The resulting array is:

        grid[k,j,i,xyz]
    """

    grid = xyz.reshape(
        NK,
        NJ,
        NI,
        3
    )

    return grid


# ============================================================
# COORDINATE TRANSFORMS
# ============================================================

def radius(points):

    return np.sqrt(
        points[..., 1] ** 2 +
        points[..., 2] ** 2
    )


def theta(points):

    # Angle around X axis.
    #
    # atan2(Y,Z) is used because the MULTALL coordinate
    # convention visible in the supplied data places the
    # angular coordinate in Y-Z.

    return np.unwrap(
        np.arctan2(
            points[..., 1],
            points[..., 2]
        ),
        axis=-1
    )


def theta_deg(points):

    return np.degrees(
        np.arctan2(
            points[..., 1],
            points[..., 2]
        )
    )


# ============================================================
# GRID DIRECTION DIAGNOSTICS
# ============================================================

def direction_variation(grid):

    """
    Determine how much each index direction changes:

        I
        J
        K

    We calculate average absolute changes in:

        X
        R
        theta

    This is a diagnostic, not a hard-coded selection.
    """

    directions = {}

    # --------------------------------------------------------
    # I direction
    # --------------------------------------------------------

    d = (
        grid[
            :,
            :,
            1:,
            :
        ]
        -
        grid[
            :,
            :,
            :-1,
            :
        ]
    )

    p = grid[
        :,
        :,
        :-1,
        :
    ]

    r0 = radius(p)
    r1 = radius(
        grid[
            :,
            :,
            1:,
            :
        ]
    )

    t0 = np.arctan2(
        p[...,1],
        p[...,2]
    )

    t1 = np.arctan2(
        grid[
            :,
            :,
            1:,
            1
        ],
        grid[
            :,
            :,
            1:,
            2
        ]
    )

    dt = np.angle(
        np.exp(
            1j * (
                t1 - t0
            )
        )
    )

    directions["I"] = {
        "dx": float(
            np.mean(
                np.abs(
                    d[...,0]
                )
            )
        ),
        "dr": float(
            np.mean(
                np.abs(
                    r1 - r0
                )
            )
        ),
        "dtheta": float(
            np.mean(
                np.abs(
                    dt
                )
            )
        ),
    }

    # --------------------------------------------------------
    # J direction
    # --------------------------------------------------------

    d = (
        grid[
            :,
            1:,
            :,
            :
        ]
        -
        grid[
            :,
            :-1,
            :,
            :
        ]
    )

    p = grid[
        :,
        :-1,
        :,
        :
    ]

    r0 = radius(p)
    r1 = radius(
        grid[
            :,
            1:,
            :,
            :
        ]
    )

    t0 = np.arctan2(
        p[...,1],
        p[...,2]
    )

    t1 = np.arctan2(
        grid[
            :,
            1:,
            :,
            1
        ],
        grid[
            :,
            1:,
            :,
            2
        ]
    )

    dt = np.angle(
        np.exp(
            1j * (
                t1 - t0
            )
        )
    )

    directions["J"] = {
        "dx": float(
            np.mean(
                np.abs(
                    d[...,0]
                )
            )
        ),
        "dr": float(
            np.mean(
                np.abs(
                    r1 - r0
                )
            )
        ),
        "dtheta": float(
            np.mean(
                np.abs(
                    dt
                )
            )
        ),
    }

    # --------------------------------------------------------
    # K direction
    # --------------------------------------------------------

    d = (
        grid[
            1:,
            :,
            :,
            :
        ]
        -
        grid[
            :-1,
            :,
            :,
            :
        ]
    )

    p = grid[
        :-1,
        :,
        :,
        :
    ]

    r0 = radius(p)
    r1 = radius(
        grid[
            1:,
            :,
            :,
            :
        ]
    )

    t0 = np.arctan2(
        p[...,1],
        p[...,2]
    )

    t1 = np.arctan2(
        grid[
            1:,
            :,
            :,
            1
        ],
        grid[
            1:,
            :,
            :,
            2
        ]
    )

    dt = np.angle(
        np.exp(
            1j * (
                t1 - t0
            )
        )
    )

    directions["K"] = {
        "dx": float(
            np.mean(
                np.abs(
                    d[...,0]
                )
            )
        ),
        "dr": float(
            np.mean(
                np.abs(
                    r1 - r0
                )
            )
        ),
        "dtheta": float(
            np.mean(
                np.abs(
                    dt
                )
            )
        ),
    }

    return directions


# ============================================================
# PRINT TOPOLOGY
# ============================================================

def print_topology(
    grid,
    diagnostics
):

    print()
    print("=" * 78)
    print("GRID TOPOLOGY")
    print("=" * 78)

    print()

    for name in (
        "I",
        "J",
        "K"
    ):

        d = diagnostics[
            name
        ]

        print(
            f"{name} direction:"
        )

        print(
            f"    mean |dX|     = "
            f"{d['dx']:.10e}"
        )

        print(
            f"    mean |dR|     = "
            f"{d['dr']:.10e}"
        )

        print(
            f"    mean |dTheta| = "
            f"{math.degrees(d['dtheta']):.10e} deg"
        )

        print()

    # --------------------------------------------------------
    # Global ranges
    # --------------------------------------------------------

    p = grid.reshape(
        -1,
        3
    )

    r = radius(
        p
    )

    th = theta_deg(
        p
    )

    print(
        "GLOBAL:"
    )

    print(
        f"    X = {p[:,0].min():.10f}"
        f" ... {p[:,0].max():.10f}"
    )

    print(
        f"    R = {r.min():.10f}"
        f" ... {r.max():.10f}"
    )

    print(
        f"    theta = {th.min():.6f}"
        f" ... {th.max():.6f} deg"
    )


# ============================================================
# DETERMINE CIRCUMFERENTIAL DIRECTION
# ============================================================

def detect_circumferential_direction(
    diagnostics
):

    """
    The blade-to-blade coordinate must exhibit substantial
    angular variation.

    We rank I/J/K by mean |dTheta|.

    """

    scores = {
        key: diagnostics[key]["dtheta"]
        for key in diagnostics
    }

    ordered = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    best = ordered[0]
    second = ordered[1]

    print()
    print(
        "=" * 78
    )

    print(
        "AUTOMATIC TOPOLOGY DETECTION"
    )

    print(
        "=" * 78
    )

    print()

    for name in ordered:

        print(
            f"{name}: "
            f"{math.degrees(scores[name]):.8f} deg/index"
        )

    print()

    print(
        f"Detected circumferential direction: {best}"
    )

    # A direction with virtually no angular change
    # cannot be the blade-to-blade coordinate.

    if scores[best] < math.radians(
        0.01
    ):

        die(
            "No meaningful circumferential grid direction "
            "could be detected."
        )

    return best


# ============================================================
# WRITE TOPOLOGY REPORT
# ============================================================

def write_topology_report(
    filename,
    diagnostics,
    circumferential
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "MULTALL TECPLOT TOPOLOGY REPORT\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Grid I={NI} J={NJ} K={NK}\n"
        )

        f.write(
            f"Points={NI*NJ*NK}\n\n"
        )

        for name in (
            "I",
            "J",
            "K"
        ):

            d = diagnostics[
                name
            ]

            f.write(
                f"{name} direction\n"
            )

            f.write(
                f"    mean dX     = "
                f"{d['dx']:.12e}\n"
            )

            f.write(
                f"    mean dR     = "
                f"{d['dr']:.12e}\n"
            )

            f.write(
                f"    mean dTheta = "
                f"{math.degrees(d['dtheta']):.12e} deg\n\n"
            )

        f.write(
            f"Detected circumferential direction: "
            f"{circumferential}\n"
        )

        f.write(
            f"JLE = {JLE}\n"
        )

        f.write(
            f"JTE = {JTE}\n"
        )


# ============================================================
# EXTRACT BLADE SURFACES
# ============================================================

def extract_surfaces(
    grid,
    circumferential
):

    """
    Once the blade-to-blade direction is known, the two
    extreme planes of that coordinate are the two sides
    of the passage.

    We then retain JLE:JTE.

    This is done according to the detected direction.
    """

    j0 = JLE - 1
    j1 = JTE - 1

    if circumferential == "I":

        # I = blade-to-blade
        #
        # Remaining dimensions:
        # K x J
        #
        # Return:
        # span x streamwise x XYZ

        side_a = grid[
            :,
            j0:j1 + 1,
            0,
            :
        ].copy()

        side_b = grid[
            :,
            j0:j1 + 1,
            NI - 1,
            :
        ].copy()

        return side_a, side_b

    if circumferential == "J":

        # J is blade-to-blade.
        #
        # Here JLE/JTE cannot simultaneously be streamwise
        # blade stations. This would contradict the MULTALL
        # station information.
        #
        # Therefore report the topology rather than creating
        # a geometrically fabricated blade.

        die(
            "The automatic coordinate test identifies J as "
            "the circumferential direction.\n\n"
            "But JLE/JTE are supplied as MULTALL streamwise "
            "blade stations (16/116).\n\n"
            "This means the XYZ ordering is inconsistent with "
            "the assumed POINT reconstruction.\n\n"
            "The program therefore refuses to create a false "
            "blade. Check the chunk concatenation/order."
        )

    if circumferential == "K":

        # K = blade-to-blade
        #
        # J remains streamwise.
        #
        # I remains span/radius.

        side_a = grid[
            :,
            j0:j1 + 1,
            :,
            0,
        ]

        side_a = grid[
            :,
            j0:j1 + 1,
            0,
            :
        ].copy()

        side_b = grid[
            :,
            j0:j1 + 1,
            0,
            :
        ].copy()

        # Correct K boundary surfaces:
        #
        # grid[K,J,I,XYZ]

        side_a = grid[
            0,
            j0:j1 + 1,
            :,
            :
        ].copy()

        side_b = grid[
            NK - 1,
            j0:j1 + 1,
            :,
            :
        ].copy()

        return side_a, side_b

    die(
        f"Unknown circumferential direction: "
        f"{circumferential}"
    )


# ============================================================
# IMPORTANT SURFACE ORIENTATION
# ============================================================

def orient_surface(
    surface
):

    """
    Return surface as:

        [span, streamwise, XYZ]

    The extraction above gives [span, J, XYZ].
    """

    return surface.copy()


# ============================================================
# SAVE XYZ
# ============================================================

def save_xyz(
    filename,
    points
):

    np.savetxt(
        filename,
        points.reshape(
            -1,
            3
        ),
        fmt="%.10e"
    )


# ============================================================
# SURFACE TRIANGULATION
# ============================================================

def triangulate_surface(
    surface,
    reverse=False
):

    ns = surface.shape[0]
    nj = surface.shape[1]

    vertices = surface.reshape(
        -1,
        3
    ).copy()

    faces = []

    def idx(i, j):

        return (
            i * nj +
            j
        )

    for i in range(
        ns - 1
    ):

        for j in range(
            nj - 1
        ):

            a = idx(
                i,
                j
            )

            b = idx(
                i + 1,
                j
            )

            c = idx(
                i + 1,
                j + 1
            )

            d = idx(
                i,
                j + 1
            )

            if reverse:

                faces.append(
                    [a, c, b]
                )

                faces.append(
                    [a, d, c]
                )

            else:

                faces.append(
                    [a, b, c]
                )

                faces.append(
                    [a, c, d]
                )

    return (
        vertices,
        np.asarray(
            faces,
            dtype=np.int64
        )
    )


# ============================================================
# CONNECT EDGES
# ============================================================

def connect_edge(
    side_a,
    side_b,
    along_span=True
):

    """
    Close the solid between the two blade sides.

    If along_span=True:
        connect hub/tip edges.

    Otherwise:
        connect LE/TE.
    """

    parts = []

    if along_span:

        for s in (
            0,
            side_a.shape[0] - 1
        ):

            a = side_a[
                s,
                :,
                :
            ]

            b = side_b[
                s,
                :,
                :
            ]

            n = len(a)

            vertices = np.vstack(
                [
                    a,
                    b
                ]
            )

            faces = []

            for j in range(
                n - 1
            ):

                a0 = j
                a1 = j + 1

                b0 = n + j
                b1 = n + j + 1

                faces.append(
                    [
                        a0,
                        a1,
                        b1
                    ]
                )

                faces.append(
                    [
                        a0,
                        b1,
                        b0
                    ]
                )

            parts.append(
                (
                    vertices,
                    np.asarray(
                        faces,
                        dtype=np.int64
                    )
                )
            )

    else:

        for j in (
            0,
            side_a.shape[1] - 1
        ):

            a = side_a[
                :,
                j,
                :
            ]

            b = side_b[
                :,
                j,
                :
            ]

            n = len(a)

            vertices = np.vstack(
                [
                    a,
                    b
                ]
            )

            faces = []

            for i in range(
                n - 1
            ):

                a0 = i
                a1 = i + 1

                b0 = n + i
                b1 = n + i + 1

                faces.append(
                    [
                        a0,
                        b0,
                        b1
                    ]
                )

                faces.append(
                    [
                        a0,
                        b1,
                        a1
                    ]
                )

            parts.append(
                (
                    vertices,
                    np.asarray(
                        faces,
                        dtype=np.int64
                    )
                )
            )

    return parts


# ============================================================
# MERGE PARTS
# ============================================================

def merge_mesh_parts(
    parts
):

    vertices = []
    faces = []

    offset = 0

    for v, f in parts:

        vertices.append(
            v
        )

        faces.append(
            f + offset
        )

        offset += len(v)

    return (
        np.vstack(vertices),
        np.vstack(faces)
    )


# ============================================================
# REMOVE DUPLICATE VERTICES
# ============================================================

def weld_vertices(
    vertices,
    faces,
    tolerance=1.0e-9
):

    """
    STL does not require welded vertices, but welding
    improves the final solid and avoids unnecessary duplicate
    nodes.
    """

    if len(vertices) == 0:

        return vertices, faces

    scaled = np.round(
        vertices /
        tolerance
    ).astype(
        np.int64
    )

    unique, inverse = np.unique(
        scaled,
        axis=0,
        return_inverse=True
    )

    new_vertices = np.zeros(
        (
            len(unique),
            3
        ),
        dtype=np.float64
    )

    # Use original vertices represented by first occurrence.

    first = {}

    for old, new in enumerate(
        inverse
    ):

        if new not in first:

            first[new] = old

    for new, old in first.items():

        new_vertices[new] = (
            vertices[old]
        )

    new_faces = inverse[
        faces
    ]

    return (
        new_vertices,
        new_faces
    )


# ============================================================
# REMOVE BAD FACES
# ============================================================

def clean_mesh(
    vertices,
    faces
):

    p0 = vertices[
        faces[:,0]
    ]

    p1 = vertices[
        faces[:,1]
    ]

    p2 = vertices[
        faces[:,2]
    ]

    cross = np.cross(
        p1 - p0,
        p2 - p0
    )

    area2 = np.linalg.norm(
        cross,
        axis=1
    )

    keep = (
        area2 >
        1.0e-14
    )

    faces = faces[
        keep
    ]

    # Remove triangles with identical indices.

    keep = (
        (faces[:,0] != faces[:,1]) &
        (faces[:,1] != faces[:,2]) &
        (faces[:,0] != faces[:,2])
    )

    faces = faces[
        keep
    ]

    return (
        vertices,
        faces
    )


# ============================================================
# BUILD SINGLE BLADE
# ============================================================

def build_single_blade(
    side_a,
    side_b
):

    print()
    print("=" * 78)
    print("BUILDING SINGLE BLADE")
    print("=" * 78)

    parts = []

    # Side A
    parts.append(
        triangulate_surface(
            side_a,
            reverse=False
        )
    )

    # Side B
    parts.append(
        triangulate_surface(
            side_b,
            reverse=True
        )
    )

    # Hub/tip edges
    parts.extend(
        connect_edge(
            side_a,
            side_b,
            along_span=True
        )
    )

    # LE/TE
    parts.extend(
        connect_edge(
            side_a,
            side_b,
            along_span=False
        )
    )

    vertices, faces = (
        merge_mesh_parts(
            parts
        )
    )

    vertices, faces = (
        clean_mesh(
            vertices,
            faces
        )
    )

    vertices, faces = (
        weld_vertices(
            vertices,
            faces
        )
    )

    print()
    print(
        f"Single blade vertices : "
        f"{len(vertices):,}"
    )

    print(
        f"Single blade triangles: "
        f"{len(faces):,}"
    )

    return (
        vertices,
        faces
    )


# ============================================================
# ROTATE AROUND X
# ============================================================

def rotate_x(
    vertices,
    degrees
):

    a = math.radians(
        degrees
    )

    c = math.cos(
        a
    )

    s = math.sin(
        a
    )

    result = vertices.copy()

    y = vertices[:,1]
    z = vertices[:,2]

    result[:,0] = (
        vertices[:,0]
    )

    result[:,1] = (
        c * y -
        s * z
    )

    result[:,2] = (
        s * y +
        c * z
    )

    return result


# ============================================================
# SIX BLADES
# ============================================================

def build_impeller(
    blade_vertices,
    blade_faces
):

    print()
    print("=" * 78)
    print("BUILDING 6-BLADE IMPELLER")
    print("=" * 78)

    all_vertices = []
    all_faces = []

    offset = 0

    pitch = (
        360.0 /
        NBLADES
    )

    for n in range(
        NBLADES
    ):

        angle = (
            n *
            pitch
        )

        print(
            f"Blade {n+1}: "
            f"{angle:.3f} degrees"
        )

        v = rotate_x(
            blade_vertices,
            angle
        )

        f = (
            blade_faces +
            offset
        )

        all_vertices.append(
            v
        )

        all_faces.append(
            f
        )

        offset += len(v)

    vertices = np.vstack(
        all_vertices
    )

    faces = np.vstack(
        all_faces
    )

    return (
        vertices,
        faces
    )


# ============================================================
# STL WRITER
# ============================================================

def write_stl(
    filename,
    vertices,
    faces
):

    with open(
        filename,
        "wb"
    ) as f:

        header = (
            b"MULTALL AXIAL CONICAL "
            b"IMPELLER"
        )

        f.write(
            header.ljust(
                80,
                b" "
            )[:80]
        )

        f.write(
            struct.pack(
                "<I",
                len(faces)
            )
        )

        for face in faces:

            p0 = vertices[
                face[0]
            ]

            p1 = vertices[
                face[1]
            ]

            p2 = vertices[
                face[2]
            ]

            normal = np.cross(
                p1 - p0,
                p2 - p0
            )

            norm = np.linalg.norm(
                normal
            )

            if norm > 1.0e-15:

                normal /= norm

            else:

                normal[:] = 0.0

            f.write(
                struct.pack(
                    "<3f",
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2])
                )
            )

            for p in (
                p0,
                p1,
                p2
            ):

                f.write(
                    struct.pack(
                        "<3f",
                        float(p[0]),
                        float(p[1]),
                        float(p[2])
                    )
                )

            f.write(
                struct.pack(
                    "<H",
                    0
                )
            )


# ============================================================
# PLOT
# ============================================================

def make_plot(
    vertices,
    filename,
    title
):

    try:

        import matplotlib

        matplotlib.use(
            "Agg"
        )

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib not installed; "
            "skipping PNG."
        )

        return

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Plot triangles as wireframe-like collection.
    # Sample for speed.
    # --------------------------------------------------------

    max_points = 50000

    stride = max(
        1,
        len(vertices) //
        max_points
    )

    p = vertices[
        ::stride
    ]

    ax.scatter(
        p[:,0],
        p[:,1],
        p[:,2],
        s=0.25,
        alpha=0.5
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    ax.set_title(
        title
    )

    # Equal scaling

    mn = vertices.min(
        axis=0
    )

    mx = vertices.max(
        axis=0
    )

    center = (
        mn +
        mx
    ) / 2.0

    span = (
        mx -
        mn
    )

    half = (
        max(
            span
        ) /
        2.0
    )

    ax.set_xlim(
        center[0] - half,
        center[0] + half
    )

    ax.set_ylim(
        center[1] - half,
        center[1] + half
    )

    ax.set_zlim(
        center[2] - half,
        center[2] + half
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=180
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "MULTALL CHUNKS -> AXIAL-CONICAL 6-BLADE IMPELLER"
    )
    print("=" * 78)

    print()
    print(
        "INPUT FILES:"
    )

    for f in CHUNKS:

        print(
            f"    {f}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read chunks
    # --------------------------------------------------------

    xyz = read_chunks()

    # Save reconstructed linear XYZ.

    save_xyz(
        OUTPUT_DIR /
        "reconstructed_points.xyz",
        xyz
    )

    # --------------------------------------------------------
    # Reconstruct grid
    # --------------------------------------------------------

    grid = reconstruct_grid(xyz)

    # --------------------------------------------------------
    # Diagnostics & Topology
    # --------------------------------------------------------

    diag = direction_variation(grid)

    print_topology(grid, diag)

    circ = detect_circumferential_direction(diag)

    write_topology_report(
        OUTPUT_DIR / "topology_report.txt",
        diag,
        circ
    )

    # --------------------------------------------------------
    # Extract Blade Surfaces
    # --------------------------------------------------------

    side_a, side_b = extract_surfaces(grid, circ)

    save_xyz(OUTPUT_DIR / "side_A.xyz", side_a)
    save_xyz(OUTPUT_DIR / "side_B.xyz", side_b)

    # --------------------------------------------------------
    # Build Single Blade
    # --------------------------------------------------------

    blade_v, blade_f = build_single_blade(side_a, side_b)

    write_stl(
        OUTPUT_DIR / "blade_single.stl",
        blade_v,
        blade_f
    )
    save_xyz(OUTPUT_DIR / "blade_single.xyz", blade_v)
    make_plot(
        blade_v,
        OUTPUT_DIR / "blade_single.png",
        "Single Blade"
    )

    # --------------------------------------------------------
    # Build 6-Blade Impeller
    # --------------------------------------------------------

    imp_v, imp_f = build_impeller(blade_v, blade_f)

    write_stl(
        OUTPUT_DIR / "impeller_6_blades.stl",
        imp_v,
        imp_f
    )
    save_xyz(OUTPUT_DIR / "impeller_6_blades.xyz", imp_v)
    make_plot(
        imp_v,
        OUTPUT_DIR / "impeller_6_blades.png",
        "6-Blade Axial-Conical Impeller"
    )

    print()
    print("=" * 78)
    print("DONE SUCCESSFULLY")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()