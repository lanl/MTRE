import os, uuid, json, random, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
import math

# ---------- unchanged prompts ----------
CIRCLE_PROMPT = "Count the circles in this image. Respond by counting them out loud, in the format: One, Two, Three, etc."
TRIANGLE_PROMPT = "Count the triangles in this image. Respond by counting them out loud, in the format: One, Two, Three, etc."

def ensure_dir(path="./data/Triangle"):
    os.makedirs(path, exist_ok=True)
    return path

def get_colors_from_colormap(name, num):
    cmap = plt.get_cmap(name)
    if num <= 1: return [cmap(0.0)]
    return [cmap(i/(num-1)) for i in range(num)]

def normalize_colors(colors, n):
    if mcolors.is_color_like(colors):
        return [colors]*n
    try:
        if len(colors) != n:
            raise ValueError(f"`colors` must be a single color or a list of length {n}; got {len(colors)}.")
        for c in colors:
            if not mcolors.is_color_like(c):
                raise ValueError(f"Invalid color: {c!r}")
        return list(colors)
    except TypeError:
        raise ValueError("`colors` must be a single Matplotlib color or an iterable of colors.")

def broadcast_scalar_or_list(x, n, name):
    if np.isscalar(x): return [float(x)]*n
    xx = list(x)
    if len(xx) != n: raise ValueError(f"`{name}` must be a scalar or length {n}; got {len(xx)}.")
    return [float(v) for v in xx]

def randomize_centers(centers, num_shifts=1, x_range=(0, 0.001), y_range=(0, 0.001)):
    batches = []
    for _ in range(num_shifts):
        dx = random.uniform(*x_range)
        dy = random.uniform(*y_range)
        batches.append([[cx+dx, cy+dy] for cx, cy in centers])
    return batches

# ---------- FIT HELPERS ----------
def _row_positions(n, step, center):
    if n <= 0: return []
    if n == 1: return [center]
    offset = (n - 1)/2.0
    return [center + (k - offset)*step for k in range(n)]

def _fit_row_horizontally(xs, left_bound, right_bound):
    """Shift (and if needed compress) xs so that min/max are within [left_bound,right_bound]."""
    if not xs: return xs
    xmin, xmax = min(xs), max(xs)
    width = right_bound - left_bound
    if len(xs) == 1:
        x = min(max(xs[0], left_bound), right_bound)
        return [x]
    if xmax - xmin <= width:
        # shift only
        shift = 0.0
        if xmin < left_bound: shift = left_bound - xmin
        if xmax + shift > right_bound: shift -= (xmax + shift - right_bound)
        return [x + shift for x in xs]
    # need to compress
    scale = width / (xmax - xmin)
    xmid = 0.5*(xmin + xmax)
    return [ (xmid + (x - xmid)*scale) for x in xs ]

def _clamp_centers_asymmetric(centers, left, right, bottom, top):
    """Clamp each (x,y) into [left,right]x[bottom,top]."""
    out = []
    for x,y in centers:
        out.append([min(max(x, left), right), min(max(y, bottom), top)])
    return out

# ---------- ALWAYS-FITS layout (two interleaved rows) ----------
def make_centers(num, rad, dist, x=0.5, y=0.6, *,
                 horiz_margin, top_margin, bottom_margin):
    """
    Build `num` centers so shapes with given margins fit.
    Spacing tries `min_dist = 2*rad + dist*rad`, but will shrink if needed.
    """
    min_dist = 2.0*rad + dist*rad
    top = (num + 1)//2
    bot = num//2

    # Compute the maximum allowed horizontal width
    left_bound  = horiz_margin
    right_bound = 1.0 - horiz_margin
    avail_width = max(0.0, right_bound - left_bound)

    # Choose the step that fits; shrink if necessary
    if top > 1:
        step_top = min(min_dist, avail_width / (top - 1) if top > 1 else min_dist)
    else:
        step_top = min_dist
    if bot > 1:
        step_bot = min(min_dist, avail_width / (bot - 1) if bot > 1 else min_dist)
    else:
        step_bot = min_dist

    xs_top = _row_positions(top, step_top, x)
    xs_bot = _row_positions(bot, step_bot, x + 0.5*min_dist if bot else x)

    # Fit them horizontally inside margins
    xs_top = _fit_row_horizontally(xs_top, left_bound, right_bound)
    xs_bot = _fit_row_horizontally(xs_bot, left_bound, right_bound)

    # Vertical fit: clamp y so both rows fit with asymmetric margins
    # Top row y must be <= 1 - top_margin; bottom row y - rad must be >= bottom_margin (for circles)
    y_top_allowed_max = 1.0 - top_margin
    y_bot_allowed_min = bottom_margin
    y = min(max(y, y_bot_allowed_min + 0.0), y_top_allowed_max)  # base y for top row
    y_bot = y - rad
    if bot:
        # If bottom row violates bottom margin, lift both rows together
        if y_bot < bottom_margin:
            delta = bottom_margin - y_bot
            y += delta
            y_bot += delta
        # If top row violates top margin, lower both rows together
        if y > 1.0 - top_margin:
            delta = y - (1.0 - top_margin)
            y -= delta
            y_bot -= delta

    centers = []
    centers.extend([[xx, y] for xx in xs_top])
    if bot:
        centers.extend([[xx, y_bot] for xx in xs_bot])

    # Safety: exact count
    if len(centers) > num:
        centers = centers[:num]
    elif len(centers) < num:
        centers.extend(centers[-1:] * (num - len(centers)))
    return centers

# ---------- DRAWING with guaranteed fit ----------
def draw_circles(dpi, size, radius, centers, colors, thickness):
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    colors = normalize_colors(colors, n)
    radii  = broadcast_scalar_or_list(radius, n, "radius")
    widths = broadcast_scalar_or_list(thickness, n, "thickness")

    h = float(size)
    fig, ax = plt.subplots(figsize=(h, h), dpi=int(dpi))
    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(0, h); ax.set_ylim(0, h); ax.axis("off")

    for (cx, cy), col, r, lw in zip(centers, colors, radii, widths):
        # clamp per-circle to guarantee fit
        margin = r
        cx = min(max(cx, margin), 1.0 - margin)
        cy = min(max(cy, margin), 1.0 - margin)
        ax.add_artist(plt.Circle((cx*h, cy*h), r*h, color=col, fill=False, linewidth=lw))

    image_id = str(uuid.uuid4())
    path = f"{ensure_dir('./data/Triangle')}/{image_id}.png"
    fig.savefig(path, bbox_inches='tight', dpi=int(dpi), pad_inches=0)
    plt.close(fig)
    return image_id

def draw_triangles(dpi, size, radius, centers, colors, thickness):
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    colors = normalize_colors(colors, n)
    radii  = broadcast_scalar_or_list(radius, n, "radius")
    widths = broadcast_scalar_or_list(thickness, n, "thickness")

    # triangle margins derived from your geometry:
    # width = r; height = sqrt(3)/2 * r
    # top offset = 2/3 * height = sqrt(3)/3 * r
    # bottom offset = 1/3 * height = sqrt(3)/6 * r
    h = float(size)
    fig, ax = plt.subplots(figsize=(h, h), dpi=int(dpi))
    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(0, h); ax.set_ylim(0, h); ax.axis("off")

    for (cx, cy), col, r, lw in zip(centers, colors, radii, widths):
        x_margin = 0.5 * r
        top_margin = (math.sqrt(3)/3.0) * r
        bottom_margin = (math.sqrt(3)/6.0) * r
        cx = min(max(cx, x_margin), 1.0 - x_margin)
        cy = min(max(cy, bottom_margin), 1.0 - top_margin)

        x_center, y_center = cx*h, cy*h
        tri_h = math.sqrt(3) * r * h / 2.0
        tri_w = r * h
        verts = [
            (x_center,              y_center + 2*tri_h/3),
            (x_center - tri_w/2.0,  y_center - tri_h/3),
            (x_center + tri_w/2.0,  y_center - tri_h/3),
        ]
        ax.add_artist(plt.Polygon(verts, closed=True, edgecolor=col, fill=False, linewidth=lw))

    image_id = str(uuid.uuid4())
    path = f"{ensure_dir('./data/Triangle')}/{image_id}.png"
    fig.savefig(path, bbox_inches='tight', dpi=int(dpi), pad_inches=0)
    plt.close(fig)
    return image_id

# ==========================
# Data generation
# ==========================
def main():
    sizes = [5]               # figure size in inches (square)
    dpi_list = [100]          # list of dpis
    num_circles_list = [9, 10, 11, 12, 13, 14]
    dists = [0.1,0.2]             # spacing factor relative to radius
    thicknesses = [1]         # line widths
    r_values = [5, 7, 10]        # rad = 0.5 / r

    metadata = {}
    total_images = 0
    for size in sizes:
        for dist in dists:
            for thickness in thicknesses:
                for dpi in dpi_list:
                    for r in r_values:
                        rad = 0.5 / r
                        for num in num_circles_list:
                            num_shifts = 25 if num % 2 == 0 else 3
                            color_modes = 2  # ['black'] * num and colormap
                            total_images += num_shifts * color_modes
    print(f"Estimated total images to generate: {total_images}")

    for size in sizes:
        for dist in dists:
            for thickness in thicknesses:
                for dpi in dpi_list:
                    for r in r_values:
                        rad = 0.5 / r
                        for num in num_circles_list:
                            base_centers = make_centers(
                                num, rad, dist, x=0.5, y=0.6,
                                horiz_margin=rad,      # left/right margin
                                top_margin=rad,        # space at top
                                bottom_margin=rad      # space at bottom
                            )

                            # you can gate extreme layouts if you want:
                            # e.g., skip if any x is outside [0,1] or bottom row goes negative
                            # if any((cx < 0 or cx > 1 for cx, _ in base_centers)) or any((cy < 0 for _, cy in base_centers)):
                            #     continue

                            # Randomize positions (more shifts for even case, like your original)
                            num_shifts = 25 if num % 2 == 0 else 3
                            randomized_batches = randomize_centers(base_centers, num_shifts=num_shifts)

                            # Two color modes: all black, and colormap
                            color_modes = [
                                ["black"] * num,
                                get_colors_from_colormap("tab10", num),
                            ]

                            for colors in color_modes:
                                for centers in randomized_batches:
                                    # img_id = draw_circles(
                                    #     dpi=dpi, size=size, radius=rad,
                                    #     centers=centers, colors=colors, thickness=thickness
                                    # )
                                    # metadata[img_id] = {
                                    #     "pid": img_id,
                                    #     "diameter": rad * 2,
                                    #     "centers": centers,
                                    #     "distance": dist,
                                    #     "dpi": dpi,
                                    #     "canvas_size": float(size),
                                    #     "linewidth": thickness,
                                    #     "colors": colors,
                                    #     "answer": num,
                                    #     "query": CIRCLE_PROMPT,
                                    # }

                                    tri_id = draw_triangles(
                                        dpi=dpi, size=size, radius=rad,
                                        centers=centers, colors=colors, thickness=thickness
                                    )
                                    metadata[tri_id] = {
                                        "pid": tri_id,
                                        "diameter": rad * 2,
                                        "centers": centers,
                                        "distance": dist,
                                        "dpi": dpi,
                                        "canvas_size": float(size),
                                        "linewidth": thickness,
                                        "colors": colors,
                                        "answer": num,
                                        "query": TRIANGLE_PROMPT,
                                    }

    print(len(metadata))
    save_dir = ensure_dir("./data/Triangle")
    with open(os.path.join(save_dir, "metadata.json"), "w") as fp:
        json.dump(metadata, fp)

if __name__ == "__main__":
    main()
