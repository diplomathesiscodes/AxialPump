#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MULTALL TECPLOT CHUNKS -> AXIAL-CONICAL 6-BLADE IMPELLER

INPUT
-----

Only:

    tecplot-input-shrinked-chunk.dat00
    tecplot-input-shrinked-chunk.dat01
    ...
    tecplot-input-shrinked-chunk.dat11

are used.

NO STAGEN FILE
NO FLOW-PASSAGE STL
NO VELOCITY DATA
NO PRESSURE DATA

MULTALL GRID
------------

    I = 46
    J = 141
    K = 46

Therefore:

    46 * 141 * 46 = 298356 XYZ points

MULTALL BLADE STATIONS
----------------------

    JLE = 16
    JTE = 116

ROTOR
-----

    Number of blades = 6
    Pitch = 60 degrees
    Rotation axis = X

The purpose of this program is to reconstruct the actual blade
geometry contained in ONE MULTALL flow passage.

The two blade surfaces are taken directly from the two sides of
the computational passage.

No artificial blade profile is created.
"""

from pathlib import Path
import math
import struct
import sys

import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

NI = 46
NJ = 141
NK = 46

JLE = 16
JTE = 116

NBLADES = 6

ROTATION_AXIS = "X"

CHUNK_PREFIX = "tecplot-input-shrinked-chunk.dat"

NUMBER_OF_CHUNKS = 12

OUTPUT_DIR = Path("impeller_output")


# ============================================================
# FILE LIST
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
# EXPECTED SIZE
# ============================================================

EXPECTED_POINTS = (
    NI * NJ * NK
)


# ============================================================
# BASIC FUNCTIONS
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


def finite_or_die(
    array,
    name
):

    if not np.isfinite(
        array
    ).all():

        die(
            f"Non-finite value found in {name}."
        )


# ============================================================
# NUMBER TEST
# ============================================================

def parse_float(token):

    """
    Convert Fortran-style floating-point values.

    Examples:

        1.234E-03
        1.234D-03
        -0.123
    """

    try:

        return float(
            token
            .replace(
                "D",
                "E"
            )
            .replace(
                "d",
                "e"
            )
        )

    except ValueError:

        return None


# ============================================================
# READ ONE CHUNK
# ============================================================

def read_chunk(
    filename
):

    """
    Read one chunk.

    IMPORTANT:

    The previous program used np.loadtxt() and assumed that every
    line contained three columns.

    That assumption is false for the supplied chunks.

    Some chunks contain single-value/header/remnant lines.

    We therefore accept ONLY complete lines containing exactly
    three finite numerical values.

    This is deliberately done independently for every chunk.
    """

    points = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:

                continue

            fields = line.split()

            # A coordinate record must contain exactly:
            #
            # X Y Z

            if len(fields) != 3:

                continue

            x = parse_float(
                fields[0]
            )

            y = parse_float(
                fields[1]
            )

            z = parse_float(
                fields[2]
            )

            if (
                x is None
                or
                y is None
                or
                z is None
            ):

                continue

            if not (
                math.isfinite(x)
                and
                math.isfinite(y)
                and
                math.isfinite(z)
            ):

                continue

            points.append(
                [
                    x,
                    y,
                    z
                ]
            )

    if not points:

        die(
            f"No XYZ coordinate records found in:\n"
            f"{filename}"
        )

    return np.asarray(
        points,
        dtype=np.float64
    )


# ============================================================
# READ ALL CHUNKS
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
                "Missing input file:\n\n"
                f"    {filename.resolve()}\n\n"
                "All 12 chunks must be in the same directory."
            )

        print(
            f"Reading {filename.name} ..."
        )

        data = read_chunk(
            filename
        )

        print(
            f"    XYZ points = "
            f"{len(data):,}"
        )

        all_points.append(
            data
        )

    xyz = np.vstack(
        all_points
    )

    print()
    print(
        f"Total points read : "
        f"{len(xyz):,}"
    )

    print(
        f"Expected          : "
        f"{EXPECTED_POINTS:,}"
    )

    if len(xyz) != EXPECTED_POINTS:

        die(
            "Incorrect number of XYZ points.\n\n"
            f"Expected: {EXPECTED_POINTS:,}\n"
            f"Found:    {len(xyz):,}\n\n"
            "The geometry has NOT been generated."
        )

    finite_or_die(
        xyz,
        "XYZ coordinates"
    )

    print()
    print(
        "COORDINATE INFORMATION"
    )
    print(
        "----------------------"
    )

    print(
        f"X range     : "
        f"{xyz[:,0].min():.10f} ... "
        f"{xyz[:,0].max():.10f}"
    )

    print(
        f"Y range     : "
        f"{xyz[:,1].min():.10f} ... "
        f"{xyz[:,1].max():.10f}"
    )

    print(
        f"Z range     : "
        f"{xyz[:,2].min():.10f} ... "
        f"{xyz[:,2].max():.10f}"
    )

    radius = np.sqrt(
        xyz[:,1] ** 2
        +
        xyz[:,2] ** 2
    )

    angle = np.degrees(
        np.arctan2(
            xyz[:,1],
            xyz[:,2]
        )
    )

    print(
        f"Radius      : "
        f"{radius.min():.10f} ... "
        f"{radius.max():.10f}"
    )

    print(
        f"Theta       : "
        f"{angle.min():.10f} ... "
        f"{angle.max():.10f}"
    )

    return xyz


# ============================================================
# RECONSTRUCT GRID
# ============================================================

def reconstruct_grid(
    xyz
):

    """
    Tecplot POINT ordering:

        I varies fastest
        J next
        K slowest

    Result:

        grid[K,J,I,XYZ]
    """

    grid = xyz.reshape(
        NK,
        NJ,
        NI,
        3
    )

    print()
    print("=" * 78)
    print("GRID RECONSTRUCTION")
    print("=" * 78)

    print()
    print(
        f"Grid shape = {grid.shape}"
    )

    return grid


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def radius_grid(
    grid
):

    return np.sqrt(
        grid[...,1] ** 2
        +
        grid[...,2] ** 2
    )


def angle_grid(
    grid
):

    return np.arctan2(
        grid[...,1],
        grid[...,2]
    )


def wrapped_angle_difference(
    a,
    b
):

    return np.angle(
        np.exp(
            1j * (
                b - a
            )
        )
    )


# ============================================================
# GRID DIRECTION DIAGNOSTICS
# ============================================================

def direction_diagnostics(
    grid
):

    radius = radius_grid(
        grid
    )

    angle = angle_grid(
        grid
    )

    diagnostics = {}

    # --------------------------------------------------------
    # I
    # --------------------------------------------------------

    dtheta = wrapped_angle_difference(
        angle[:,:,:-1],
        angle[:,:,1:]
    )

    dr = (
        radius[:,:,1:]
        -
        radius[:,:,:-1]
    )

    dx = (
        grid[:,:,1:,0]
        -
        grid[:,:,:-1,0]
    )

    diagnostics["I"] = {
        "dtheta":
            float(
                np.mean(
                    np.abs(dtheta)
                )
            ),
        "dr":
            float(
                np.mean(
                    np.abs(dr)
                )
            ),
        "dx":
            float(
                np.mean(
                    np.abs(dx)
                )
            )
    }

    # --------------------------------------------------------
    # J
    # --------------------------------------------------------

    dtheta = wrapped_angle_difference(
        angle[:,:-1,:],
        angle[:,1:,:]
    )

    dr = (
        radius[:,1:,:]
        -
        radius[:,:-1,:]
    )

    dx = (
        grid[:,1:,:,0]
        -
        grid[:,:-1,:,0]
    )

    diagnostics["J"] = {
        "dtheta":
            float(
                np.mean(
                    np.abs(dtheta)
                )
            ),
        "dr":
            float(
                np.mean(
                    np.abs(dr)
                )
            ),
        "dx":
            float(
                np.mean(
                    np.abs(dx)
                )
            )
    }

    # --------------------------------------------------------
    # K
    # --------------------------------------------------------

    dtheta = wrapped_angle_difference(
        angle[:-1,:,:],
        angle[1:,:,:]
    )

    dr = (
        radius[1:,:,:]
        -
        radius[:-1,:,:]
    )

    dx = (
        grid[1:,:, :,0]
        -
        grid[:-1,:,:,0]
    )

    diagnostics["K"] = {
        "dtheta":
            float(
                np.mean(
                    np.abs(dtheta)
                )
            ),
        "dr":
            float(
                np.mean(
                    np.abs(dr)
                )
            ),
        "dx":
            float(
                np.mean(
                    np.abs(dx)
                )
            )
    }

    return diagnostics


# ============================================================
# PRINT TOPOLOGY
# ============================================================

def print_diagnostics(
    diagnostics
):

    print()
    print("=" * 78)
    print("GRID DIRECTION DIAGNOSTICS")
    print("=" * 78)

    for name in (
        "I",
        "J",
        "K"
    ):

        d = diagnostics[
            name
        ]

        print()
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


# ============================================================
# DETERMINE BLADE-TO-BLADE DIRECTION
# ============================================================

def determine_blade_to_blade_direction(
    diagnostics
):

    """
    J is fixed as the MULTALL streamwise coordinate because
    JLE/JTE are supplied as J stations.

    Therefore the blade-to-blade coordinate must be I or K.

    We compare I and K angular variation.

    The direction with the larger coherent angular change is
    selected.
    """

    i_score = diagnostics[
        "I"
    ]["dtheta"]

    k_score = diagnostics[
        "K"
    ]["dtheta"]

    print()
    print("=" * 78)
    print("BLADE-TO-BLADE DIRECTION")
    print("=" * 78)

    print()
    print(
        f"I angular variation = "
        f"{math.degrees(i_score):.10f} deg/index"
    )

    print(
        f"K angular variation = "
        f"{math.degrees(k_score):.10f} deg/index"
    )

    if i_score > k_score:

        direction = "I"

    else:

        direction = "K"

    print()
    print(
        f"Selected blade-to-blade direction: "
        f"{direction}"
    )

    return direction


# ============================================================
# EXTRACT THE TWO BLADE FACES
# ============================================================

def extract_blade_faces(
    grid,
    blade_to_blade
):

    """
    J is streamwise.

    JLE=16
    JTE=116

    Python:

        j0 = 15
        j1 = 115

    The two extreme surfaces of the blade-to-blade
    computational direction are used as the actual blade
    pressure/suction surfaces.
    """

    j0 = JLE - 1
    j1 = JTE - 1

    if blade_to_blade == "K":

        # grid[K,J,I,XYZ]
        #
        # K = blade-to-blade
        # J = streamwise
        # I = span

        side_a = grid[
            0,
            j0:j1+1,
            :,
            :
        ].copy()

        side_b = grid[
            NK-1,
            j0:j1+1,
            :,
            :
        ].copy()

    elif blade_to_blade == "I":

        # grid[K,J,I,XYZ]
        #
        # I = blade-to-blade
        # J = streamwise
        # K = span

        side_a = grid[
            :,
            j0:j1+1,
            0,
            :
        ].copy()

        side_b = grid[
            :,
            j0:j1+1,
            NI-1,
            :
        ].copy()

    else:

        die(
            f"Unknown blade-to-blade direction: "
            f"{blade_to_blade}"
        )

    finite_or_die(
        side_a,
        "blade side A"
    )

    finite_or_die(
        side_b,
        "blade side B"
    )

    print()
    print("=" * 78)
    print("BLADE SURFACE EXTRACTION")
    print("=" * 78)

    print()
    print(
        f"JLE = {JLE}"
    )

    print(
        f"JTE = {JTE}"
    )

    print(
        f"Side A shape = {side_a.shape}"
    )

    print(
        f"Side B shape = {side_b.shape}"
    )

    return (
        side_a,
        side_b
    )


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
# TRIANGULATE SURFACE
# ============================================================

def triangulate_surface(
    surface,
    reverse=False
):

    """
    surface:

        [span, streamwise, XYZ]
    """

    ns = surface.shape[0]
    nj = surface.shape[1]

    vertices = surface.reshape(
        -1,
        3
    ).copy()

    faces = []

    def index(
        i,
        j
    ):

        return (
            i * nj
            +
            j
        )

    for i in range(
        ns - 1
    ):

        for j in range(
            nj - 1
        ):

            a = index(
                i,
                j
            )

            b = index(
                i+1,
                j
            )

            c = index(
                i+1,
                j+1
            )

            d = index(
                i,
                j+1
            )

            if reverse:

                faces.append(
                    [
                        a,
                        c,
                        b
                    ]
                )

                faces.append(
                    [
                        a,
                        d,
                        c
                    ]
                )

            else:

                faces.append(
                    [
                        a,
                        b,
                        c
                    ]
                )

                faces.append(
                    [
                        a,
                        c,
                        d
                    ]
                )

    return (
        vertices,
        np.asarray(
            faces,
            dtype=np.int64
        )
    )


# ============================================================
# EDGE CONNECTION
# ============================================================

def connect_span_edge(
    side_a,
    side_b,
    span_index
):

    """
    Connect corresponding streamwise curves.

    span_index:

        0            -> hub-side edge
        last         -> tip-side edge
    """

    a = side_a[
        span_index,
        :,
        :
    ]

    b = side_b[
        span_index,
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

    return (
        vertices,
        np.asarray(
            faces,
            dtype=np.int64
        )
    )


# ============================================================
# LE / TE CONNECTION
# ============================================================

def connect_stream_edge(
    side_a,
    side_b,
    j_index
):

    """
    Connect the two blade faces at a streamwise station.

    j_index:

        0       -> LE
        last    -> TE
    """

    a = side_a[
        :,
        j_index,
        :
    ]

    b = side_b[
        :,
        j_index,
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

    return (
        vertices,
        np.asarray(
            faces,
            dtype=np.int64
        )
    )


# ============================================================
# MERGE MESH PARTS
# ============================================================

def merge_parts(
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
# WELD VERTICES
# ============================================================

def weld_vertices(
    vertices,
    faces,
    tolerance=1.0e-10
):

    """
    Remove duplicated vertices generated where surfaces and
    closing edges meet.

    This is important for producing a proper closed STL.
    """

    if len(vertices) == 0:

        return (
            vertices,
            faces
        )

    key = np.round(
        vertices /
        tolerance
    ).astype(
        np.int64
    )

    unique_key, inverse = np.unique(
        key,
        axis=0,
        return_inverse=True
    )

    new_vertices = np.zeros(
        (
            len(unique_key),
            3
        ),
        dtype=np.float64
    )

    first = {}

    for old_index, new_index in enumerate(
        inverse
    ):

        if new_index not in first:

            first[
                new_index
            ] = old_index

    for new_index, old_index in first.items():

        new_vertices[
            new_index
        ] = vertices[
            old_index
        ]

    new_faces = inverse[
        faces
    ]

    return (
        new_vertices,
        new_faces
    )


# ============================================================
# REMOVE DEGENERATE TRIANGLES
# ============================================================

def clean_faces(
    vertices,
    faces
):

    if len(faces) == 0:

        return (
            vertices,
            faces
        )

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
        p1-p0,
        p2-p0
    )

    area2 = np.linalg.norm(
        cross,
        axis=1
    )

    good = (
        area2 >
        1.0e-14
    )

    faces = faces[
        good
    ]

    good = (
        (faces[:,0] != faces[:,1])
        &
        (faces[:,1] != faces[:,2])
        &
        (faces[:,0] != faces[:,2])
    )

    faces = faces[
        good
    ]

    return (
        vertices,
        faces
    )


# ============================================================
# BUILD ONE BLADE
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

    # --------------------------------------------------------
    # Surface A
    # --------------------------------------------------------

    parts.append(
        triangulate_surface(
            side_a,
            reverse=False
        )
    )

    # --------------------------------------------------------
    # Surface B
    # --------------------------------------------------------

    parts.append(
        triangulate_surface(
            side_b,
            reverse=True
        )
    )

    # --------------------------------------------------------
    # Hub-side edge
    # --------------------------------------------------------

    parts.append(
        connect_span_edge(
            side_a,
            side_b,
            0
        )
    )

    # --------------------------------------------------------
    # Tip-side edge
    # --------------------------------------------------------

    parts.append(
        connect_span_edge(
            side_a,
            side_b,
            side_a.shape[0]-1
        )
    )

    # --------------------------------------------------------
    # Leading edge
    # --------------------------------------------------------

    parts.append(
        connect_stream_edge(
            side_a,
            side_b,
            0
        )
    )

    # --------------------------------------------------------
    # Trailing edge
    # --------------------------------------------------------

    parts.append(
        connect_stream_edge(
            side_a,
            side_b,
            side_a.shape[1]-1
        )
    )

    vertices, faces = merge_parts(
        parts
    )

    vertices, faces = clean_faces(
        vertices,
        faces
    )

    vertices, faces = weld_vertices(
        vertices,
        faces
    )

    vertices, faces = clean_faces(
        vertices,
        faces
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
# ROTATE ABOUT X
# ============================================================

def rotate_about_x(
    vertices,
    degrees
):

    angle = math.radians(
        degrees
    )

    c = math.cos(
        angle
    )

    s = math.sin(
        angle
    )

    result = vertices.copy()

    y = vertices[:,1]
    z = vertices[:,2]

    result[:,1] = (
        c*y
        -
        s*z
    )

    result[:,2] = (
        s*y
        +
        c*z
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

    vertices = []
    faces = []

    pitch = (
        360.0 /
        NBLADES
    )

    offset = 0

    for blade_number in range(
        NBLADES
    ):

        angle = (
            blade_number *
            pitch
        )

        print(
            f"Blade {blade_number+1}: "
            f"{angle:.3f} degrees"
        )

        v = rotate_about_x(
            blade_vertices,
            angle
        )

        f = (
            blade_faces
            +
            offset
        )

        vertices.append(
            v
        )

        faces.append(
            f
        )

        offset += len(v)

    vertices = np.vstack(
        vertices
    )

    faces = np.vstack(
        faces
    )

    return (
        vertices,
        faces
    )


# ============================================================
# STL WRITER
# ============================================================

def write_binary_stl(
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
                p1-p0,
                p2-p0
            )

            length = np.linalg.norm(
                normal
            )

            if length > 1.0e-15:

                normal /= length

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

            f.write(
                struct.pack(
                    "<3f",
                    float(p0[0]),
                    float(p0[1]),
                    float(p0[2])
                )
            )

            f.write(
                struct.pack(
                    "<3f",
                    float(p1[0]),
                    float(p1[1]),
                    float(p1[2])
                )
            )

            f.write(
                struct.pack(
                    "<3f",
                    float(p2[0]),
                    float(p2[1]),
                    float(p2[2])
                )
            )

            f.write(
                struct.pack(
                    "<H",
                    0
                )
            )


# ============================================================
# SAVE PLY-LIKE TRIANGLE DATA
# ============================================================

def save_mesh_xyz(
    filename,
    vertices
):

    np.savetxt(
        filename,
        vertices,
        fmt="%.10e"
    )


# ============================================================
# OPTIONAL PLOT
# ============================================================

def plot_mesh(
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
            "matplotlib is not installed; "
            "PNG diagnostic skipped."
        )

        return

    fig = plt.figure(
        figsize=(10,8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Avoid excessive plotting memory.

    maximum_points = 60000

    stride = max(
        1,
        len(vertices) //
        maximum_points
    )

    p = vertices[
        ::stride
    ]

    ax.scatter(
        p[:,0],
        p[:,1],
        p[:,2],
        s=0.15,
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

    # Equal visual scale.

    minimum = vertices.min(
        axis=0
    )

    maximum = vertices.max(
        axis=0
    )

    center = (
        minimum +
        maximum
    ) / 2.0

    span = (
        maximum -
        minimum
    )

    half = (
        max(span)
        /
        2.0
    )

    ax.set_xlim(
        center[0]-half,
        center[0]+half
    )

    ax.set_ylim(
        center[1]-half,
        center[1]+half
    )

    ax.set_zlim(
        center[2]-half,
        center[2]+half
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
# TOPOLOGY REPORT
# ============================================================

def write_report(
    filename,
    diagnostics,
    blade_to_blade
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "MULTALL AXIAL-CONICAL IMPELLER\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"I = {NI}\n"
        )

        f.write(
            f"J = {NJ}\n"
        )

        f.write(
            f"K = {NK}\n\n"
        )

        f.write(
            f"Expected points = "
            f"{EXPECTED_POINTS}\n\n"
        )

        f.write(
            f"JLE = {JLE}\n"
        )

        f.write(
            f"JTE = {JTE}\n\n"
        )

        f.write(
            f"Number of blades = "
            f"{NBLADES}\n"
        )

        f.write(
            "Rotation axis = X\n\n"
        )

        f.write(
            f"Blade-to-blade direction = "
            f"{blade_to_blade}\n\n"
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
                f"    dX = "
                f"{d['dx']:.12e}\n"
            )

            f.write(
                f"    dR = "
                f"{d['dr']:.12e}\n"
            )

            f.write(
                f"    dTheta = "
                f"{math.degrees(d['dtheta']):.12e} deg\n\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "MULTALL TECPLOT -> "
        "AXIAL-CONICAL IMPELLER"
    )
    print("=" * 78)

    print()
    print(
        "Input:"
    )

    for filename in CHUNKS:

        print(
            f"    {filename}"
        )

    print()
    print(
        f"Grid = "
        f"{NI} x {NJ} x {NK}"
    )

    print(
        f"Expected points = "
        f"{EXPECTED_POINTS:,}"
    )

    print(
        f"Blades = {NBLADES}"
    )

    print(
        "Axis = X"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------