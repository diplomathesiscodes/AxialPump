import re
import matplotlib.pyplot as plt
import numpy as np

with open("stagen.out", "r") as f:
    text = f.read()

# ------------------------------------------------------------
# Extract PRE-STACKING table
# ------------------------------------------------------------

pattern_before = re.compile(
    r"J,\s*XGRID,\s*YGRID,\s*RGRID(.*?)(?=\n\s*\n|\n\s*THE BLADE CENTROID)",
    re.S | re.IGNORECASE,
)

m = pattern_before.search(text)
if not m:
    raise RuntimeError("Could not find pre-stacking grid.")

before = []
# Matches integer indices as well as scientific/standard floats
num_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")

for line in m.group(1).splitlines():
    vals = num_pattern.findall(line)
    if len(vals) >= 4:
        j, x, y, r = vals[:4]
        before.append([int(float(j)), float(x), float(y), float(r)])

before = np.array(before)

# ------------------------------------------------------------
# Extract AFTER-STACKING table
# ------------------------------------------------------------

pattern_after = re.compile(
    r"AXIAL & RADIAL COORDINATES ON THE SS AFTER STACKING(.*)",
    re.S | re.IGNORECASE,
)

m = pattern_after.search(text)
if not m:
    raise RuntimeError("Could not find post-stacking grid.")

after = []
for line in m.group(1).splitlines():
    vals = num_pattern.findall(line)
    if len(vals) >= 3:
        try:
            j, x, r = vals[:3]
            after.append([int(float(j)), float(x), float(r)])
        except ValueError:
            pass

after = np.array(after)

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

j_before = before[:, 0].astype(int)
x_before = before[:, 1]
y_before = before[:, 2]
r_before = before[:, 3]

j_after = after[:, 0].astype(int)
x_after = after[:, 1]
r_after = after[:, 2]

# Match J values safely using dictionaries for O(N) lookup
before_dict = {int(row[0]): row[1:] for row in before}
after_dict = {int(row[0]): row[1:] for row in after}

common = np.intersect1d(j_before, j_after)

xb = np.array([before_dict[j][0] for j in common])
yb = np.array([before_dict[j][1] for j in common])
rb = np.array([before_dict[j][2] for j in common])

xa = np.array([after_dict[j][0] for j in common])
ra = np.array([after_dict[j][1] for j in common])

dx = xa - xb

print("Number of grid points:", len(common))
print("Mean stacking shift:", np.mean(dx))
print("Minimum shift:", np.min(dx))
print("Maximum shift:", np.max(dx))
print("Std. deviation:", np.std(dx))

print(
    "\nMaximum radial error before stacking:", np.max(np.abs(rb - 0.5))
)
print("Maximum radial error after stacking:", np.max(np.abs(ra - 0.5)))

# ------------------------------------------------------------
# Plot X-Y geometry
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(x_before, y_before, "b.-", label="Before stacking")

# Plot common points to ensure X and Y length alignment
plt.plot(xa, yb, "r--", label="After stacking (Translated X)")

plt.scatter(
    0.021974165,
    -0.02389,
    color="black",
    s=80,
    zorder=5,
    label="Blade centroid",
)

plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Blade Section Before/After Stacking")
plt.show()

# ------------------------------------------------------------
# Plot stacking displacement
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(common, dx, "k.-")
plt.axhline(
    np.mean(dx),
    color="red",
    linestyle="--",
    label=f"Mean = {np.mean(dx):.8e}",
)

plt.xlabel("J")
plt.ylabel("X after - X before")
plt.grid(True)
plt.legend()
plt.title("Stacking Displacement")
plt.show()
