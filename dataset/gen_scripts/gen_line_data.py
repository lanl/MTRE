#!/usr/bin/env python3
"""
Line Intersection Dataset Generator

Generates synthetic images of two colored lines on a grid, with varying numbers
of intersection points. Used for evaluating VLM counting/spatial reasoning.

Usage:
    cd MTRE_R
    mkdir -p ./data/Lines
    python dataset/gen_scripts/gen_line_data.py

Output:
    ./data/Lines/
    ├── gt_0_image_*.png   # Images with 0 intersections
    ├── gt_1_image_*.png   # Images with 1 intersection
    ├── gt_2_image_*.png   # Images with 2 intersections
    └── metadata.json      # Ground truth labels and metadata

The script generates 300 total images (100 per intersection count: 0, 1, 2),
each rendered at multiple line thicknesses (2, 4) for robustness testing.

Based on: https://arxiv.org/abs/2407.06581 (VLMs Are Blind)
"""
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from numbers import Rational
import random
import json
from matplotlib import cm
from matplotlib import colormaps
import matplotlib.colors as mcolors
from tqdm import tqdm
from PIL import Image
import itertools
import copy
from collections import defaultdict
from PIL import Image, ImageDraw

LINE_PROMPT = 'Question: How many intersection points are there?'
def orientation(p, q, r):
    """Return the orientation of the triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def on_segment(p, q, r):
    """Check if point r lies on line segment pq."""
    if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
        return True
    return False

def lines_intersect(p1, q1, p2, q2):
    """Check if line segment p1q1 intersects with p2q2."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q1, q2):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, q2, p1):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q2, q1):
        return True

    return False

def count_intersections(line_segments):
    """Count the number of intersections between a list of line segments."""
    count = 0
    n = len(line_segments)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1, q1 = line_segments[i]
            p2, q2 = line_segments[j]
            if lines_intersect(p1, q1, p2, q2):
                count += 1
                
    return count

def convert_fig_to_pil(fig):
    # Draw the figure on the canvas
    fig.canvas.draw()

    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()

    # Get the RGB buffer from the figure
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    # Create a PIL image from the RGB buffer
    image = Image.fromarray(buf)

    return image
grid_size = 12
cell_size = 1 / grid_size

ticks = []

for i in range(grid_size):
    edge = round(i * cell_size, 2)
    if i == 0:
        edge += 0.1 * cell_size

    ticks.append(edge)

ticks.append(1.0 - 0.1 * cell_size)

left_edge = ticks[0]
right_edge = ticks[-1]
mid_edge = ticks[grid_size//2]


line_dict = defaultdict(int)

visited = []
metadata = dict()

cnt = 0

while np.sum(list(line_dict.values())) != 300:
    first_line = []
    second_line = []


    y_init_1, y_init_2 = random.sample(ticks, 2)
    first_line.append((left_edge, y_init_1))
    second_line.append((left_edge, y_init_2))

    dist_1 = abs(y_init_2 - y_init_1) // cell_size


    while True:
        y_mid_1, y_mid_2 = random.sample(ticks, 2)
        if (y_mid_1 != right_edge or y_init_1 != right_edge) and (y_mid_1 != left_edge or y_init_1 != left_edge) and (y_mid_2 != right_edge or y_init_2 != right_edge) and (y_mid_2 != left_edge or y_init_2 != left_edge):
            break
            

    first_line.append((mid_edge, y_mid_1))
    second_line.append((mid_edge, y_mid_2))

    dist_2 = abs(y_mid_2 - y_mid_1) // cell_size


    while True:
        y1, y2 = random.sample(ticks, 2)
        if (y_mid_1 != right_edge or y1 != right_edge) and (y_mid_1 != left_edge or y1 != left_edge) and (y_mid_2 != right_edge or y2 != right_edge) and (y_mid_2 != left_edge or y2 != left_edge):
            break

    first_line.append((right_edge, y1))
    second_line.append((right_edge, y2))

    dist_3 = abs(y2 - y1) // cell_size

    if dist_1 == 0:
        dist_1 = 1
    if dist_2 == 0:
        dist_2 = 1
    if dist_3 == 0:
        dist_3 = 1





    line_segments = []

    p1, q1 = first_line[0], first_line[1]
    p2, q2 = second_line[0], second_line[1]
    line_segments = [(p1, q1), (p2, q2)] 

    num_intersections = count_intersections(line_segments)

    line_segments = []
    p1, q1 = first_line[1], first_line[2]
    p2, q2 = second_line[1], second_line[2]
    line_segments = [(p1, q1), (p2, q2)] 

    num_intersections += count_intersections(line_segments)
    # print(num_intersections)


    if line_dict[str(num_intersections)] != 100 and (first_line not in visited or second_line not in visited):

        visited.append(first_line)
        visited.append(second_line)

        line_dict[str(num_intersections)] += 1



        data1 = [(round(5 * x, 2), round(5 * y, 2)) for (x, y) in first_line]

        data2 = [(round(5 * x, 2), round(5 * y, 2)) for (x, y) in second_line]

        for dpi in [300]:#[100, 200, 300]:
            for linewidth in [2, 4]:
                fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
                x_coords, y_coords = zip(*data1)
                ax.plot(x_coords, y_coords, linestyle='solid', color='b', linewidth=linewidth)
                x_coords, y_coords = zip(*data2)
                ax.plot(x_coords, y_coords, linestyle='solid', color='r', linewidth=linewidth)
                plt.ylim(0, 5)
                plt.xlim(0, 5)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.grid(color='gray', linestyle='-', linewidth=1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.set_aspect('equal', adjustable='box')
                ax.axis('off')
                if dpi == 100:
                    pix_size = 384
                elif dpi == 200:
                    pix_size = 768
                else:
                    pix_size = 1152

                plt.tight_layout(pad=0.0)
                image = convert_fig_to_pil(fig)
                image = image.resize((pix_size, pix_size))
                image.save("./data/Lines/gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth) + '_resolution_' + str(pix_size) + '.png')
            
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)] = dict()
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]['pid'] = "gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["answer"] = num_intersections
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["linewidth"] = linewidth
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["query"] = LINE_PROMPT
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["left"] = linewidth
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["resolution"] = dpi
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["distances"] = [dist_1, dist_2, dist_3]
                metadata["gt_" + str(num_intersections) + '_image_' + str(cnt) + '_thickness_' + str(linewidth)+ '_resolution_' + str(pix_size)]["grid_size"] = grid_size

                plt.close(fig)

        cnt += 1
        print(cnt)

print(len(metadata))

with open("./data/Lines/metadata.json", 'w') as fp:
    json.dump(metadata, fp)