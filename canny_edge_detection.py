import argparse
import os
import time

import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--image_dir", required=True, help="path to the directory containing the images"
)
ap.add_argument(
    "-o",
    "--output_dir",
    required=True,
    help="path to the directory where the edge detected images will be saved",
)
ap.add_argument(
    "-k", "--kernel_size", default=5, help="Size of the kernel to be used by the Sobel filter"
)
args = vars(ap.parse_args())

# Get the paths to the input and output directories
image_dir = args["image_dir"]
output_dir = args["output_dir"]
kernel_size = int(args["kernel_size"])

# Create the output directory if it does not already exist
os.makedirs(output_dir, exist_ok=True)

total_time = 0
# Loop through all the images in the input directory
for filename in os.listdir(image_dir):
    # Load the image
    image = cv2.imread(os.path.join(image_dir, filename))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    # Apply Sobel edge detection
    # filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 100, 150)

    end_time = time.time()
    total_time += end_time - start_time
    # Save the edge detected image to the output directory
    cv2.imwrite(os.path.join(output_dir, filename), edges)

total_images = len(os.listdir(image_dir))
average_time = total_time / total_images
print(f"Total time: {total_time}")
print(f"Total images: {total_images}")
print(f"Average time per image: {average_time}")
