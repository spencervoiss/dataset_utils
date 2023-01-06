import argparse
import os
import sys
import time

import cv2

FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)
ALGORITHM_ORIGIN = (20, 40)
PARAM_ORIGINS = [(20, 60), (20, 80), (20, 100)]


def downscale_image(image):
    """Downscales the input image to half its original size"""
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return downscaled_image


parser = argparse.ArgumentParser(
    description="Interactive application for visualizing a denoising and edge detection pipeline"
)
parser.add_argument(
    "-i",
    "--input_dir",
    help="Path to directory containing the images to be filtered",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    help="Optional argument for saving the filtered images to a particular directory",
)
args = parser.parse_args()

images_dir = args.input_dir
try:
    images = os.listdir(images_dir)
except:
    print("Images directory is empty!", file=sys.stderr)
    exit(1)

cb_image_trackbar = lambda val: val
cb_denoise_algorithm_trackbar = lambda val: val
cb_denoise_trackbar_1 = lambda val: val
cb_denoise_trackbar_2 = lambda val: val
cb_denoise_trackbar_3 = lambda val: val

image_trackbar_name = "Image #"
denoise_algorithm_trackbar_name = "Denoising Alg. #"
denoise_trackbar_1_name = "Param 1"
denoise_trackbar_2_name = "Param 2"
denoise_trackbar_3_name = "Param 3"
denoise_window_name = "Denoised Image"
denoise_algorithms = {
    "No Denoising": [],
    "Gaussian Blur": ["Kernel Size", "Sigma"],
    "Median Filter": ["Kernel Size"],
    "Bilateral Filter": ["Kernel Size", "Pixel Intensity Threshold", "Distance Threshold"],
    "Non-Local Means Denoising": ["Strentgh", "Template Window Size", "Search Window Size"],
}
denoise_algorithm_names = list(denoise_algorithms.keys())

cv2.namedWindow(denoise_window_name)
cv2.createTrackbar(image_trackbar_name, denoise_window_name, 0, len(images) - 1, cb_image_trackbar)
cv2.createTrackbar(
    denoise_algorithm_trackbar_name,
    denoise_window_name,
    0,
    len(denoise_algorithms) - 1,
    cb_denoise_algorithm_trackbar,
)
cv2.createTrackbar(denoise_trackbar_1_name, denoise_window_name, 1, 51, cb_denoise_trackbar_1)
cv2.createTrackbar(denoise_trackbar_2_name, denoise_window_name, 1, 300, cb_denoise_trackbar_2)
cv2.createTrackbar(denoise_trackbar_3_name, denoise_window_name, 1, 300, cb_denoise_trackbar_3)

image_idx = cb_image_trackbar(0)
denoise_algorithm_idx = cb_denoise_algorithm_trackbar(0)
denoise_param_1 = cb_denoise_trackbar_1(1)
denoise_param_2 = cb_denoise_trackbar_2(1)
denoise_param_3 = cb_denoise_trackbar_3(1)

cb_edges_algorithm_trackbar = lambda val: val
cb_edges_trackbar_1 = lambda val: val
cb_edges_trackbar_2 = lambda val: val
cb_edges_trackbar_3 = lambda val: val

edges_algorithm_trackbar_name = "Edge Detection Alg. #"
edges_trackbar_1_name = "Param 1"
edges_trackbar_2_name = "Param 2"
edges_trackbar_3_name = "Param 3"
edges_window_name = "Detected Edges"
edge_detection_algorithms = [
    "No Edge Detection",
    "Canny Edge Detection",
    "Sobel Edge Detection",
    "Laplacian Edge Detection",
]

cv2.namedWindow(edges_window_name)
cv2.createTrackbar(
    edges_algorithm_trackbar_name,
    edges_window_name,
    0,
    len(edge_detection_algorithms) - 1,
    cb_edges_algorithm_trackbar,
)
cv2.createTrackbar(edges_trackbar_1_name, edges_window_name, 1, 51, cb_edges_trackbar_1)
cv2.createTrackbar(edges_trackbar_2_name, edges_window_name, 1, 300, cb_edges_trackbar_2)
cv2.createTrackbar(edges_trackbar_3_name, edges_window_name, 1, 300, cb_edges_trackbar_3)

edges_algorithm_idx = cb_edges_algorithm_trackbar(0)
edges_param_1 = cb_edges_trackbar_1(1)
edges_param_2 = cb_edges_trackbar_2(1)
edges_param_3 = cb_edges_trackbar_3(1)

key = cv2.waitKey(1)
while key != 113:
    image_idx = cv2.getTrackbarPos(image_trackbar_name, denoise_window_name)

    denoise_algorithm_idx = cv2.getTrackbarPos(denoise_algorithm_trackbar_name, denoise_window_name)
    denoise_param_1 = cv2.getTrackbarPos(denoise_trackbar_1_name, denoise_window_name)
    denoise_param_2 = cv2.getTrackbarPos(denoise_trackbar_2_name, denoise_window_name)
    denoise_param_3 = cv2.getTrackbarPos(denoise_trackbar_3_name, denoise_window_name)

    edges_algorithm_idx = cv2.getTrackbarPos(edges_algorithm_trackbar_name, edges_window_name)
    edges_param_1 = cv2.getTrackbarPos(edges_trackbar_1_name, edges_window_name)
    edges_param_2 = cv2.getTrackbarPos(edges_trackbar_2_name, edges_window_name)
    edges_param_3 = cv2.getTrackbarPos(edges_trackbar_3_name, edges_window_name)

    image_path = os.path.join(images_dir, images[image_idx])
    image_full = cv2.imread(image_path)
    image = downscale_image(image_full)
    image_filtered = image.copy()

    denoise_algorithm_name = denoise_algorithm_names[denoise_algorithm_idx]
    denoise_param_names = denoise_algorithms[denoise_algorithm_name]
    image_filtered_drawn = cv2.putText(
        image_filtered.copy(),
        denoise_algorithm_name,
        ALGORITHM_ORIGIN,
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )
    for idx, param in enumerate(denoise_param_names):
        image_filtered_drawn = cv2.putText(
            image_filtered_drawn,
            f"Param {idx + 1}: {param}",
            PARAM_ORIGINS[idx],
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
    cv2.imshow(denoise_window_name, image_filtered_drawn)

    image_edges = image_filtered.copy()
    edges_algorithm_name = edge_detection_algorithms[edges_algorithm_idx]
    image_edges_drawn = cv2.putText(
        image_edges.copy(),
        edges_algorithm_name,
        ALGORITHM_ORIGIN,
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )
    cv2.imshow(edges_window_name, image_edges_drawn)
    key = cv2.waitKey(1)

cv2.destroyAllWindows()
