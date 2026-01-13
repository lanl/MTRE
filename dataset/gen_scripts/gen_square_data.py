#!/usr/bin/env python3
"""
Nested Squares Dataset Generator

Generates synthetic images of nested/concentric squares for counting tasks.
Used for evaluating VLM counting and spatial reasoning capabilities.

Usage:
    cd MTRE_R
    mkdir -p ./data/Squares
    python dataset/gen_scripts/gen_square_data.py

Output:
    ./data/Squares/
    ├── *.png              # Generated square images
    └── metadata.json      # Ground truth labels

Based on: https://arxiv.org/abs/2407.06581 (VLMs Are Blind)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import json

SQUARE_PROMPT = 'How many nested squares are there?  Respond by counting them out loud, in the format: One, Two, Three, etc.'
def compute_squares(center, size, depth, reduction_factor, padding, squares_list):
    if depth == 0:
        return

    # Store the current square's details
    squares_list.append({"center": center, "size": size})

    # Calculate the size of the next square, reduced by the reduction factor and padding
    new_size = size * reduction_factor - padding

    # Ensure new_size is positive
    if new_size <= 0:
        return

    # Generate random offsets within bounds to ensure no overlap, adjusted for padding
    max_offset = (size - new_size - padding) / 2
    offset_x = random.uniform(-max_offset, max_offset)
    offset_y = random.uniform(-max_offset, max_offset)

    # Calculate the new center
    new_center = (center[0] + offset_x, center[1] + offset_y)

    # Recursive call to compute further nested squares
    compute_squares(
        new_center, new_size, depth - 1, reduction_factor, padding, squares_list
    )


def plot_squares(ax, squares_list, line_thickness):
    for square in squares_list:
        center = square["center"]
        size = square["size"]
        # Create and add a square patch to the axes
        square_patch = patches.Rectangle(
            (center[0] - size / 2, center[1] - size / 2),
            size,
            size,
            fill=False,
            linewidth=line_thickness,
        )
        ax.add_patch(square_patch)


def generate_and_save_images(num_images_per_depth, depths, save_path):
    os.makedirs(save_path, exist_ok=True)
    image_details = []
    metadata = dict()

    for depth in depths:
        for i in range(num_images_per_depth):
            center = (random.uniform(-5, 5), random.uniform(-5, 5))
            initial_size = random.uniform(8, 18)
            reduction_factor = 0.75
            padding = 0.75

            # Compute all squares first
            squares_list = []
            compute_squares(
                center, initial_size, depth, reduction_factor, padding, squares_list
            )

            # Plot and save images with different line thicknesses
            for line_thickness in [2, 3, 4]:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_aspect("equal")
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
                ax.axis("off")

                plot_squares(ax, squares_list, line_thickness)

                image_name = f"nested_squares_depth_{depth}_image_{i+1}_thickness_{line_thickness}.pdf"
                plt.savefig(os.path.join(save_path, image_name), format="pdf")
                # save png
                image_name_png = f"nested_squares_depth_{depth}_image_{i+1}_thickness_{line_thickness}.png"
                image_name_no_png = f"nested_squares_depth_{depth}_image_{i+1}_thickness_{line_thickness}"
                plt.savefig(os.path.join(save_path, image_name_png), format="png")
                plt.close(fig)
                ii = image_name_no_png
                metadata[image_name_no_png] = dict()
                metadata[ii] = dict()
                metadata[ii]["pid"] = image_name_no_png
                metadata[ii]["answer"] = depth
                metadata[ii]["query"] = SQUARE_PROMPT

                # image_details.append(
                #     {
                #         "pid": image_name,
                #         "depth": depth,
                #         "center": center,
                #         "initial_size": initial_size,
                #         "reduction_factor": reduction_factor,
                #         "line_thickness": line_thickness,
                #         "padding": padding,
                #         "squares": squares_list,
                #     }
                # )

    # Save details to a JSON file
    print(len(metadata))
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# Example usage: generate 50 images for each depth ranging from 2 to 5
generate_and_save_images(90, [2, 3, 4, 5, 6], "./data/Squares")