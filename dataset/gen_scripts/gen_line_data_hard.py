#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, random, argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# General helpers
# -----------------------------
def ensure_dir(path): 
    os.makedirs(path, exist_ok=True); 
    return path

def convert_fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(buf)

def make_ticks(grid_size: int, edge_margin_frac: float = 0.1):
    """Grid ticks in [0,1] with a small margin away from 0 and 1."""
    cell = 1.0 / grid_size
    ticks = []
    for i in range(grid_size):
        edge = i * cell
        if i == 0: edge += edge_margin_frac * cell
        ticks.append(round(edge, 6))
    ticks.append(round(1.0 - edge_margin_frac * cell, 6))
    return ticks

def parse_images_per_k(s: str, max_k: int):
    """
    Parse "0:100,1:80,2:60" → {0:100,1:80,2:60}.
    Any k not listed defaults to 0. Valid k ∈ [0, max_k].
    """
    out = {k: 0 for k in range(max_k + 1)}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part: 
            continue
        k_str, n_str = part.split(":")
        k = int(k_str.strip())
        n = int(n_str.strip())
        if not (0 <= k <= max_k):
            raise ValueError(f"k={k} out of range; valid 0..{max_k}")
        out[k] = n
    return out

# -----------------------------
# Core geometry
# -----------------------------
def sample_pair_with_k_intersections(
    k: int,
    ticks,
    xs,
    *,
    margin: float = 0.0,
):
    """
    Build two polylines with x-positions `xs` (len(xs) >= 2).
    Exactly k intersections across len(xs)-1 windows by flipping the
    sign of (y2 - y1) across exactly k windows.
    """
    assert len(xs) >= 2, "Need at least two x positions."
    max_k = len(xs) - 1
    assert 0 <= k <= max_k, f"k must be in [0, {max_k}]"

    usable = list(ticks)
    if margin > 0:
        lo, hi = usable[0], usable[-1]
        usable = [t for t in usable if (lo + margin) < t < (hi - margin)]
        if len(usable) < 2:
            usable = list(ticks)  # fallback

    # choose k windows (indices 0..len(xs)-2) where order flips
    windows = list(range(len(xs) - 1))
    flip_windows = set(random.sample(windows, k))

    # s[j] = sign of (y2 - y1) at column j; start +1 at left, flip in chosen windows
    s = [+1]
    for j in range(1, len(xs)):
        s.append(-s[-1] if (j - 1) in flip_windows else s[-1])

    y1, y2 = [], []
    for j in range(len(xs)):
        a, b = random.sample(usable, 2)
        lo, hi = (a, b) if a < b else (b, a)
        if s[j] > 0:
            y1.append(lo); y2.append(hi)  # y1 < y2
        else:
            y1.append(hi); y2.append(lo)  # y1 > y2

    first_line  = [(xs[j], y1[j]) for j in range(len(xs))]
    second_line = [(xs[j], y2[j]) for j in range(len(xs))]
    return first_line, second_line

def measure_k_general(first_line, second_line):
    """
    Count intersections across all windows by sign flips of (y2 - y1) per column.
    This matches how we constructed the pairs.
    """
    ydiff = [second_line[j][1] - first_line[j][1] for j in range(len(first_line))]
    def sgn(x, eps=1e-12):
        if x >  eps: return  1
        if x < -eps: return -1
        return 0  # identical y (shouldn’t happen due to random.sample)
    signs = [sgn(v) for v in ydiff]
    k = 0
    for j in range(len(signs) - 1):
        if signs[j] == 0 or signs[j+1] == 0:
            # extremely rare; treat equal as no flip
            continue
        if signs[j] != signs[j+1]:
            k += 1
    return k

def dedup_key(first_line, second_line, prec=6):
    """Hashable key to avoid duplicate geometry."""
    def norm(poly): return tuple((round(x,prec), round(y,prec)) for x,y in poly)
    return (norm(first_line), norm(second_line))

# -----------------------------
# Rendering
# -----------------------------
def draw_pair(
    first_line,
    second_line,
    *,
    figsize_in=5.0,
    dpi=300,
    linewidth=2,
    grid=False,
    grid_size=12,
    out_pixels=None,
    colors=("b", "r"),
):
    """Draw two polylines to a square PIL image."""
    s = figsize_in
    def scale(poly): return [(s*x, s*y) for x,y in poly]
    p1, p2 = scale(first_line), scale(second_line)

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, s); ax.set_ylim(0, s); ax.axis('off')

    x1, y1 = zip(*p1); x2, y2 = zip(*p2)
    ax.plot(x1, y1, color=colors[0], linewidth=linewidth)
    ax.plot(x2, y2, color=colors[1], linewidth=linewidth)

    if grid:
        ticks = [s * t for t in np.linspace(0, 1, grid_size + 1)]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.grid(color='gray', linestyle='-', linewidth=1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout(pad=0.0)
    img = convert_fig_to_pil(fig)
    plt.close(fig)
    if out_pixels:
        img = img.resize((out_pixels, out_pixels), Image.BICUBIC)
    return img

# -----------------------------
# Main driver
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate line-intersection images with arbitrary k (0..columns-1).")
    parser.add_argument("--outdir", type=str, default="./data/Lines")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--columns", type=int, default=7,
                        help="Number of vertical columns (points per polyline). Max k = columns-1.")
    parser.add_argument("--figsize", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pixels", type=int, default=1152)
    parser.add_argument("--line-widths", type=int, nargs="+", default=[2,4])
    parser.add_argument("--images-per-k", type=str, default="0:50,1:50,2:50,3:50,4:50,5:50,6:50",
                        help='Comma list like "0:100,1:80,2:60". Valid k up to columns-1.')
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--with-grid", action="store_true", help="Draw background grid.")
    parser.add_argument("--margin", type=float, default=0.0, help="Optional y-margin inside [0,1].")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    outdir = ensure_dir(args.outdir)

    # Build y-ticks and x-positions
    ticks = make_ticks(args.grid_size, edge_margin_frac=0.1)
    # Choose evenly spaced indices over ticks for x columns
    idxs = np.linspace(0, len(ticks)-1, args.columns, dtype=int)
    xs = [ticks[i] for i in idxs]  # strictly increasing
    max_k = len(xs) - 1

    # Parse quotas (k up to max_k)
    targets = parse_images_per_k(args.images_per_k, max_k=max_k)
    total_needed = sum(targets.values())
    print(f"Columns: {args.columns} → max k = {max_k}")
    print(f"Targets per k: {targets} (total base pairs: {total_needed})")

    counts = defaultdict(int)
    visited = set()
    metadata = {}
    counter = 0

    while sum(counts.values()) < total_needed:
        # pick a k that still needs images
        remaining = [k for k in range(max_k+1) if counts[k] < targets[k]]
        if not remaining:
            break
        k = random.choice(remaining)

        # sample pair and verify
        first_line, second_line = sample_pair_with_k_intersections(
            k, ticks, xs, margin=args.margin
        )
        key = dedup_key(first_line, second_line)
        if key in visited:
            continue
        k_measured = measure_k_general(first_line, second_line)
        if k_measured != k:   # ultra-rare; be conservative
            continue

        visited.add(key)
        counts[k] += 1

        for lw in args.line_widths:
            img = draw_pair(
                first_line, second_line,
                figsize_in=args.figsize, dpi=args.dpi, linewidth=lw,
                grid=args.with_grid, grid_size=args.grid_size, out_pixels=args.pixels
            )
            name = f"gt_{k}_image_{counter}_thickness_{lw}_resolution_{args.pixels}"
            img.save(os.path.join(outdir, f"{name}.png"))

            metadata[name] = {
                "pid": name,
                "answer": k,
                "linewidth": lw,
                "query": "Question: How many intersection points are there? Respond by counting them out loud, in the format: One, Two, Three, etc.",
                "resolution": args.dpi,
                "grid_size": args.grid_size,
                "columns": args.columns,
                "xs": xs,
                "poly1": first_line,
                "poly2": second_line,
            }
        counter += 1

        if counter % 25 == 0:
            made = sum(counts.values())
            print(f"Progress: {made}/{total_needed} base pairs (k dist: " +
                  ", ".join(f"{i}:{counts[i]}" for i in range(max_k+1)) + ")")

    # Save metadata
    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    print(f"Done. Saved {len(metadata)} images (multiple widths per pair).")
    print("Final counts:", {i: counts[i] for i in range(max_k+1)})
    print("Metadata:", meta_path)

if __name__ == "__main__":
    main()
