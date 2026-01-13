#!/usr/bin/env python3
"""
Olympic-Like Logo Dataset Generator

Generates synthetic images of interlocking rings (Olympic logo style) for
counting tasks. Used for evaluating VLM counting with overlapping shapes.

Usage:
    cd MTRE_R
    mkdir -p ./data/OlympicLikeLogo
    python dataset/gen_scripts/gen_olympic_data.py

Output:
    ./data/OlympicLikeLogo/
    ├── *.png              # Generated ring images
    └── metadata.json      # Ground truth labels

Based on: https://arxiv.org/abs/2407.06581 (VLMs Are Blind)
"""
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from numbers import Rational
import random
import json
from matplotlib import cm

CIRCLE_PROMPT = 'Count the circles in this image. Respond by counting them out loud, in the format: One, Two, Three, etc.'
TRIANGLE_PROMPT = 'Count the triangles in this image. Respond by counting them out loud, in the format: One, Two, Three, etc.'

def hue_to_rgb(hue):
    rgb = hsv_to_rgb([hue, 1, 1])
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
     

def get_colors_from_colormap(colormap_name, num_colors):
    colormap = cm.get_cmap(colormap_name, num_colors)
    colors = [colormap(i) for i in range(num_colors)]
    return colors

import random

# Function to randomize circle positions
def randomize_centers(centers, num_shifts=1, x_range=(0, .001), y_range=(0, .001)):
    randomized_centers = []
    for _ in range(num_shifts):
        shift_x = random.uniform(*x_range)
        shift_y = random.uniform(*y_range)
        randomized_centers.append([[cx + shift_x, cy + shift_y] for cx, cy in centers])
    return randomized_centers     

import uuid
# def draw_circles(dpi, size, radius, centers, colors, thickness):

#     assert len(centers) == len(colors)
#     h=5
#     fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim(0, h)
#     ax.set_ylim(0, h)
#     ax.axis("off")

#     for center, color in zip(centers, colors):
#       circle1_plot = plt.Circle((center[0] * h, center[1] * h), radius * h, color=color, fill=False, linewidth=thickness)
#       ax.add_artist(circle1_plot)

#     ii = str(uuid.uuid4())
#     fig.savefig("/usr/projects/unsupgan/MM_TDA/LVLM-LP/data/OlympicLikeLogo/"+ ii + '.png', bbox_inches='tight', dpi=dpi, pad_inches=0)
#     # plt.show()
#     plt.close(fig)
#     return ii
    # return image_id

def draw_triangles(dpi, size, radius, centers, colors, thickness):
    assert len(centers) == len(colors), "Centers and colors must have the same length."
    
    h = size  # Plot size, h is the plot size (width and height)
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(h, h), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')  # Equal scaling
    ax.set_xlim(0, h)
    ax.set_ylim(0, h)
    ax.axis("off")  # Hide axes
    
    # Plot each triangle
    for center, color in zip(centers, colors):
        # Adjust the center position to fit within the limits (multiplying by h)
        x_center, y_center = center[0] * h, center[1] * h
        
        # Define the size of the triangle (equilateral) using the given radius
        # Generate the vertices of an equilateral triangle
        triangle_height = np.sqrt(3) * radius * h / 2  # Height of the equilateral triangle
        triangle_width = radius * h  # Base of the equilateral triangle
        
        # Vertices of the triangle (equilateral)
        vertices = [
            (x_center, y_center + 2 * triangle_height / 3),  # top vertex
            (x_center - triangle_width / 2, y_center - triangle_height / 3),  # bottom left
            (x_center + triangle_width / 2, y_center - triangle_height / 3)   # bottom right
        ]
        
        # Create the triangle using a Polygon
        triangle = plt.Polygon(vertices, closed=True, edgecolor=color, fill=False, linewidth=thickness)
        ax.add_artist(triangle)
    
    # Generate unique filename using UUID
    image_id = str(uuid.uuid4())
    save_path = f"./data/OlympicLikeLogo/{image_id}.png"
    # save_path = 'test_triangle.png'
    
    # Save the image without extra padding
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)  # Close the figure after saving
    
    return image_id

def draw_circles(dpi, size, radius, centers, colors, thickness):
    assert len(centers) == len(colors), "Centers and colors must have the same length."

    h = size  # Plot size, h is the plot size (width and height)
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(h, h), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')  # Equal scaling
    ax.set_xlim(0, h)
    ax.set_ylim(0, h)
    ax.axis("off")  # Hide axes
    
    # Plot each circle
    for center, color in zip(centers, colors):
        # Adjust the center position to fit within the limits (multiplying by h)
        circle = plt.Circle(
            (center[0] * h, center[1] * h),  # center of circle
            radius * h,                      # radius of circle
            color=color,                      # circle color
            fill=False,                       # no fill (can be set to True for filled circles)
            linewidth=thickness               # line thickness
        )
        ax.add_artist(circle)

    # Generate unique filename using UUID
    image_id = str(uuid.uuid4())
    save_path = f"./data/OlympicLikeLogo/{image_id}.png"
    
    # Save the image without extra padding
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)  # Close the figure after saving
    
    return image_id   

sizes = [5]
dpi = [300]

num_circles = [3, 4]
# num_circles = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]


dists = [.1]

metadata = dict()
for size in sizes:
  for dist in dists:
    for thickness in [1]:
      for d in dpi:
        for r in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: #5, 10, 15 #[2, 3, 4, 5, 6, 7, 8, 9, 10]
          rad = 0.5 / r
          for num in num_circles:
             # Loop for each pattern and randomize positions
            for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                if num % 2 != 0:
                    centers = []
                    row_1 = (num + 1) // 2
                    row_2 = row_1 - 1

                    y = 0.6
                    x = 0.5

                    ratio = dist * rad
                    min_dist = rad * 2.0 + ratio

                    if row_1 * rad * 2 + row_2 * ratio >= 1:
                        continue

                    if row_1 == 3:
                        centers.append([x, y])
                        centers.append([x - min_dist, y])
                        centers.append([x + min_dist, y])
                        centers.append([x - rad - ratio / 2, y - rad])
                        centers.append([x + rad + ratio / 2, y - rad])

                    elif row_1 == 5:
                        centers.append([x, y])
                        centers.append([x - min_dist, y])
                        centers.append([x + min_dist, y])
                        centers.append([x - 2 * min_dist, y])
                        centers.append([x + 2 * min_dist, y])
                        centers.append([x - rad - ratio / 2, y - rad])
                        centers.append([x + rad + ratio / 2, y - rad])
                        centers.append([x - rad - ratio - min_dist, y - rad])
                        centers.append([x + rad + ratio + min_dist, y - rad])
                    elif row_1 == 2:
                        centers.append([x - rad - ratio / 2, y])
                        centers.append([x + rad + ratio / 2, y])
                        centers.append([x, y - rad])
                    else:
                        centers.append([x - rad - ratio / 2, y])
                        centers.append([x + rad + ratio / 2, y])
                        centers.append([x - rad - ratio / 2 - min_dist, y])
                        centers.append([x + rad + ratio / 2 + min_dist, y])
                        centers.append([x, y - rad])
                        centers.append([x + min_dist, y - rad])
                        centers.append([x - min_dist, y - rad])

                    # Randomize positions n times
                    randomized_centers = randomize_centers(centers, num_shifts=25)

                    for randomized_center in randomized_centers:
                        ii = draw_circles(d, size, rad, randomized_center, colors, thickness)
                        metadata[ii] = dict()
                        metadata[ii]["pid"] = ii
                        metadata[ii]["diameter"] = rad * 2
                        metadata[ii]["centera"] = randomized_center
                        metadata[ii]["distance"] = dist
                        metadata[ii]["dpi"] = d
                        metadata[ii]["canvas_size"] = 5.0
                        metadata[ii]["linewidth"] = thickness
                        metadata[ii]["colors"] = colors
                        metadata[ii]["answer"] = num
                        metadata[ii]["query"] = CIRCLE_PROMPT
                        # ii = draw_triangles(d, size, rad, randomized_center, colors, thickness)
                        # metadata[ii] = dict()
                        # metadata[ii]["pid"] = ii
                        # metadata[ii]["diameter"] = rad * 2
                        # metadata[ii]["centers"] = randomized_center
                        # metadata[ii]["distance"] = dist
                        # metadata[ii]["dpi"] = d
                        # metadata[ii]["canvas_size"] = 5.0
                        # metadata[ii]["linewidth"] = thickness
                        # metadata[ii]["colors"] = colors
                        # metadata[ii]["answer"] = num
                        # metadata[ii]["query"] = TRIANGLE_PROMPT
                else:
                    row_1 = num // 2
                    row_2 = row_1

                    y = 0.6
                    x = 0.5

                    ratio = dist * rad
                    min_dist = rad * 2.0 + ratio

                    if row_2 * min_dist + 2 * rad >= 1:
                        continue

                    for i in range(2):
                        centers = []
                        if row_1 == 3:
                            centers.append([x, y])
                            centers.append([x - min_dist, y])
                            centers.append([x + min_dist, y])
                            centers.append([x - rad - ratio / 2, y - rad])
                            centers.append([x + rad + ratio / 2, y - rad])
                            if i == 0:
                                centers.append([x - rad - ratio - min_dist, y - rad])
                            else:
                                centers.append([x + rad + ratio + min_dist, y - rad])

                        elif row_1 == 2:
                            centers.append([x - rad - ratio / 2, y])
                            centers.append([x + rad + ratio / 2, y])
                            centers.append([x, y - rad])
                            if i == 0:
                                centers.append([x + min_dist, y - rad])
                            else:
                                centers.append([x - min_dist, y - rad])
                        else:
                            centers.append([x - rad - ratio / 2, y])
                            centers.append([x + rad + ratio / 2, y])
                            centers.append([x - rad - ratio / 2 - min_dist, y])
                            centers.append([x + rad + ratio / 2 + min_dist, y])
                            centers.append([x, y - rad])
                            centers.append([x + min_dist, y - rad])
                            centers.append([x - min_dist, y - rad])
                            if i == 0:
                                centers.append([x + 2 * min_dist, y - rad])
                            else:
                                centers.append([x - 2 * min_dist, y - rad])

                        # Randomize positions n times
                        randomized_centers = randomize_centers(centers, num_shifts=25)

                        for randomized_center in randomized_centers:
                            ii = draw_circles(d, size, rad, randomized_center, colors, thickness)
                            metadata[ii] = dict()
                            metadata[ii]["pid"] = ii
                            metadata[ii]["diameter"] = rad * 2
                            metadata[ii]["centers"] = randomized_center
                            metadata[ii]["distance"] = dist
                            metadata[ii]["dpi"] = d
                            metadata[ii]["canvas_size"] = 5.0
                            metadata[ii]["linewidth"] = thickness
                            metadata[ii]["colors"] = colors
                            metadata[ii]["answer"] = num
                            metadata[ii]["query"] = CIRCLE_PROMPT
                            # ii = draw_triangles(d, size, rad, randomized_center, colors, thickness)
                            # metadata[ii] = dict()
                            # metadata[ii]["pid"] = ii
                            # metadata[ii]["diameter"] = rad * 2
                            # metadata[ii]["centers"] = randomized_center
                            # metadata[ii]["distance"] = dist
                            # metadata[ii]["dpi"] = d
                            # metadata[ii]["canvas_size"] = 5.0
                            # metadata[ii]["linewidth"] = thickness
                            # metadata[ii]["colors"] = colors
                            # metadata[ii]["answer"] = num
                            # metadata[ii]["query"] = TRIANGLE_PROMPT
      
            # for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
            # for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:


            #   if num % 2 != 0:
            #     centers = []
            #     row_1 = (num + 1) // 2
            #     row_2 = row_1 - 1

            #     y = 0.6
            #     x = 0.5

            #     ratio = dist * rad
            #     min_dist = rad * 2.0 + ratio

            #     if row_1 * rad * 2 + row_2 * ratio >= 1:
            #       continue


            #     if row_1 == 3:
            #       centers.append([x, y])
            #       centers.append([x - min_dist, y])
            #       centers.append([x + min_dist, y])
            #       centers.append([x - rad - ratio/2, y - rad])
            #       centers.append([x + rad + ratio/2, y - rad])

            #     elif row_1 == 5:
            #       centers.append([x, y])
            #       centers.append([x - min_dist, y])
            #       centers.append([x + min_dist, y])
            #       centers.append([x - 2 * min_dist, y])
            #       centers.append([x + 2 * min_dist, y])
            #       centers.append([x - rad - ratio / 2, y - rad])
            #       centers.append([x + rad + ratio / 2, y - rad])
            #       centers.append([x - rad - ratio - min_dist, y - rad])
            #       centers.append([x + rad + ratio + min_dist, y - rad])
            #     elif row_1 == 2:
            #       centers.append([x - rad - ratio/2, y])
            #       centers.append([x + rad + ratio/2, y])
            #       centers.append([x, y - rad])
            #     else:
            #       centers.append([x - rad - ratio/2, y])
            #       centers.append([x + rad + ratio/2, y])
            #       centers.append([x - rad - ratio/2 - min_dist, y])
            #       centers.append([x + rad + ratio/2 + min_dist, y])
            #       centers.append([x, y - rad])
            #       centers.append([x + min_dist, y - rad])
            #       centers.append([x - min_dist, y - rad])

            #     ii = draw_circles(d, size, rad, centers, colors, thickness)
            #     metadata[ii] = dict()
            #     metadata[ii]["pid"] = ii
            #     metadata[ii]["diameter"] = rad * 2
            #     metadata[ii]["centera"] = centers
            #     metadata[ii]["distance"] = dist
            #     metadata[ii]["dpi"] = d
            #     metadata[ii]["canvas_size"] = 5.0
            #     metadata[ii]["linewidth"] = thickness
            #     metadata[ii]["colors"] = colors
            #     metadata[ii]["answer"] = num
            #     metadata[ii]["query"] = CIRCLE_PROMPT
            #   else:
            #     row_1 = num // 2
            #     row_2 = row_1

            #     y = 0.6
            #     x = 0.5

            #     ratio = dist * rad
            #     min_dist = rad * 2.0 + ratio

            #     if row_2 * min_dist + 2 * rad >= 1:
            #       continue
            #     for i in range(2):
            #       centers = []
            #       if row_1 == 3:
            #         centers.append([x, y])
            #         centers.append([x - min_dist, y])
            #         centers.append([x + min_dist, y])
            #         centers.append([x - rad - ratio/2, y - rad])
            #         centers.append([x + rad + ratio/2, y - rad])
            #         if i == 0:
            #           centers.append([x - rad - ratio - min_dist, y - rad])
            #         else:
            #           centers.append([x + rad + ratio + min_dist, y - rad])

            #       elif row_1 == 2:
            #         centers.append([x - rad - ratio/2, y])
            #         centers.append([x + rad + ratio/2, y])
            #         centers.append([x, y - rad])
            #         if i == 0:
            #           centers.append([x + min_dist, y - rad])
            #         else:
            #           centers.append([x - min_dist, y - rad])
            #       else:
            #         centers.append([x - rad - ratio/2, y])
            #         centers.append([x + rad + ratio/2, y])
            #         centers.append([x - rad - ratio/2 - min_dist, y])
            #         centers.append([x + rad + ratio/2 + min_dist, y])
            #         centers.append([x, y - rad])
            #         centers.append([x + min_dist, y - rad])
            #         centers.append([x - min_dist, y - rad])
            #         if i == 0:
            #           centers.append([x + 2 * min_dist, y - rad])
            #         else:
            #           centers.append([x - 2 * min_dist, y - rad])

            #       ii = draw_circles(d, size, rad, centers, colors, thickness)
            #       metadata[ii] = dict()
            #       metadata[ii]["pid"] = ii
            #       metadata[ii]["diameter"] = rad * 2
            #       metadata[ii]["centers"] = centers
            #       metadata[ii]["distance"] = dist
            #       metadata[ii]["dpi"] = d
            #       metadata[ii]["canvas_size"] = 5.0
            #       metadata[ii]["linewidth"] = thickness
            #       metadata[ii]["colors"] = colors
            #       metadata[ii]["answer"] = num
            #       metadata[ii]["query"] = CIRCLE_PROMPT

print(len(metadata))

with open("./data/OlympicLikeLogo/metadata.json", "w") as fp:
  json.dump(metadata, fp)
