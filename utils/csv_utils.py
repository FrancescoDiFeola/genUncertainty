import os
import csv

ROOT_DIR = "/Users/francescodifeola/Desktop/val"   # <-- change this
A_DIR = os.path.join(ROOT_DIR, "A")
B_DIR = os.path.join(ROOT_DIR, "B")

OUTPUT_CSV = "/Users/francescodifeola/Desktop/val.csv"

# --------------------------------------------------
# Collect all B files for fast lookup
# --------------------------------------------------
B_files = set()
for root, _, files in os.walk(B_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            rel_path = os.path.relpath(os.path.join(root, f), B_DIR)
            B_files.add(rel_path)

pairs = []

# --------------------------------------------------
# Walk through A and find matching B
# --------------------------------------------------
for root, _, files in os.walk(A_DIR):
    for f in files:
        if not f.lower().endswith(".png"):
            continue

        A_full = os.path.join(root, f)
        A_rel = os.path.relpath(A_full, A_DIR)

        # Expected B filename (_ref added)
        B_rel = A_rel.replace("_rgb_anon.png", "_ref_rgb_anon.png")

        if B_rel in B_files:
            pairs.append((
                os.path.join("A", A_rel),
                os.path.join("B", B_rel)
            ))

print(f"Found {len(pairs)} matching pairs")

# --------------------------------------------------
# Write CSV (NO HEADER)
# --------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pairs)

print(f"CSV saved to: {OUTPUT_CSV}")


############ in this version the names of the images match only partially ##########
import os
import csv
import re
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT_DIR = "/Users/francescodifeola/Desktop/test"   # <-- change this
A_DIR = os.path.join(ROOT_DIR, "A")
B_DIR = os.path.join(ROOT_DIR, "B")

OUTPUT_CSV = "/Users/francescodifeola/Desktop/test.csv"

frame_re = re.compile(r"(frame_\d{6})")

pairs = []

# --------------------------------------------------
# Iterate over subfolders (e.g. GOPR0364)
# --------------------------------------------------
for sub in os.listdir(A_DIR):
    A_sub = os.path.join(A_DIR, sub)
    B_sub = os.path.join(B_DIR, sub)

    if not os.path.isdir(A_sub) or not os.path.isdir(B_sub):
        continue

    # Index B files by frame
    B_by_frame = {}
    for f in os.listdir(B_sub):
        if f.endswith(".png"):
            m = frame_re.search(f)
            if m:
                B_by_frame[m.group(1)] = f

    # Match A files
    for f in os.listdir(A_sub):
        if not f.endswith(".png"):
            continue

        m = frame_re.search(f)
        if not m:
            continue

        frame_id = m.group(1)

        if frame_id in B_by_frame:
            A_rel = os.path.join("A", sub, f)
            B_rel = os.path.join("B", sub, B_by_frame[frame_id])
            pairs.append((A_rel, B_rel))

print(f"Found {len(pairs)} matching pairs")

# --------------------------------------------------
# Write CSV (NO HEADER)
# --------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pairs)

print(f"CSV saved to: {OUTPUT_CSV}")