import argparse
import os
import sys
import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)
ALGORITHM_ORIGIN = (20, 40)
PARAM_ORIGINS = [(20, 60), (20, 80), (20, 100)]
TIME_ORIGIN = (20, 120)
LEGEND_TOP_LEFT = (10, 10)
LEGEND_BOTTOM_RIGHT = (350, 140)
LEGEND_BG_COLOR = (255, 255, 255)


def downscale_image(image):
    """Downscales the input image to half its original size"""
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return downscaled_image


def kernel_size_err():
    """Prints an error stating that the kernel size must be an odd number"""
    print("Kernel size must be an even number. Using unfiltered image", file=sys.stderr)


def denoise_image(image: np.ndarray, algorithm_name: str, *args):
    """Selects the appropriate denoising algorithm from the algorithm name
    and performs denoising on the input image.

    Args:
        image (cv2.Mat): Image to be denoised
        algorithm_name (str): Name of the algorithm to be performed on image
        *args (ints): Parameters for the denoising algorithm
    """
    if algorithm_name == "No Denoising":
        return image.copy()

    elif algorithm_name == "Gaussian Blur":
        kernel_size = args[0]
        sigma = args[1]
        gaussian = image.copy()
        try:
            gaussian = cv2.GaussianBlur(gaussian, (kernel_size, kernel_size), sigma)
        except:
            kernel_size_err()
        return gaussian

    elif algorithm_name == "Median Filter":
        kernel_size = args[0]
        median = image.copy()
        try:
            median = cv2.medianBlur(median, kernel_size)
        except:
            kernel_size_err()
        return median

    elif algorithm_name == "Bilateral Filter":
        kernel_size = args[0]
        pixel_intensity_threshold = args[1]
        spatial_distance_threshold = args[2]
        bilateral = image.copy()
        try:
            bilateral = cv2.bilateralFilter(
                bilateral, kernel_size, pixel_intensity_threshold, spatial_distance_threshold
            )
        except Exception as err:
            print(err, file=sys.stderr)
        return bilateral

    elif algorithm_name == "Non-Local Means Denoising":
        h = args[0]
        template_window_size = args[1]
        search_window_size = args[2]
        nlmeans = image.copy()
        try:
            nlmeans = cv2.fastNlMeansDenoising(
                nlmeans,
                h=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
            )
        except Exception as err:
            print(err, file=sys.stderr)
        return nlmeans

    return image.copy()


def draw_legend_background(image: np.ndarray):
    """Draws a gray rectangle at the top left corner of the image as the background for a legend"""
    cv2.rectangle(
        image,
        LEGEND_TOP_LEFT,
        LEGEND_BOTTOM_RIGHT,
        LEGEND_BG_COLOR,
        thickness=-1,
        lineType=cv2.LINE_8,
    )


def draw_legend(image: np.ndarray, algorithm_name: str, time: float, param_names):
    image_legend = image.copy()
    draw_legend_background(image_legend)
    legend = cv2.putText(
        image_legend,
        algorithm_name,
        ALGORITHM_ORIGIN,
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )
    for idx, param in enumerate(param_names):
        legend = cv2.putText(
            image_legend,
            f"Param {idx + 1}: {param}",
            PARAM_ORIGINS[idx],
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
    image_legend = cv2.putText(
        image_legend,
        f"Time: {time:.3f}",
        TIME_ORIGIN,
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )
    return image_legend


def detect_edges(image: np.ndarray, algorithm_name: str, *args):
    """Performs the desired edge detection algorithm on image with parameters as defined by args

    Args:
    image (np.ndarray): Image for edge detection to be performed on
    algorithm_name (str): Name of edge detection algorithm
    *args: Parameters for the edge detection of choice
    """
    if algorithm_name == "No Edge Detection":
        return image.copy()

    elif algorithm_name == "Canny Edge Detection":
        canny = image.copy()
        low_threshold = args[0]
        high_threshold = args[1]
        try:
            canny = cv2.Canny(canny, low_threshold, high_threshold)
        except Exception as err:
            print(err, file=sys.stdout)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    elif algorithm_name == "Sobel Edge Detection":
        sobel = image.copy()
        kernel_size = args[0]
        dx_order = args[1]
        dy_order = args[2]
        try:
            sobel = cv2.Sobel(sobel, cv2.CV_64F, dx_order, dy_order, ksize=kernel_size)
        except Exception as err:
            print(err, file=sys.stderr)
        return sobel

    elif algorithm_name == "Laplacian Edge Detection":
        laplacian = image.copy()
        kernel_size = args[0]
        try:
            laplacian = cv2.Laplacian(laplacian, cv2.CV_64F, ksize=kernel_size)
        except Exception as err:
            print(err, file=sys.stderr)
        return laplacian

    return image.copy()


parser = argparse.ArgumentParser(
    description="Interactive application for visualizing a denoising and edge detection pipeline"
)
parser.add_argument(
    "-i",
    "--input_dir",
    help="Path to directory containing the images to be denoised",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    help="Optional argument for saving the denoised images to a particular directory",
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
cb_denoise_trackbar_0 = lambda val: val
cb_denoise_trackbar_1 = lambda val: val
cb_denoise_trackbar_2 = lambda val: val

image_trackbar_name = "Image #"
denoise_algorithm_trackbar_name = "Denoising Alg. #"
denoise_trackbar_0_name = "Param 1"
denoise_trackbar_1_name = "Param 2"
denoise_trackbar_2_name = "Param 3"
denoise_window_name = "Denoised Image"
denoise_algorithms = {
    "No Denoising": [],
    "Gaussian Blur": ["Kernel Size (must be odd)", "Sigma"],
    "Median Filter": ["Kernel Size (must be odd)"],
    "Bilateral Filter": [
        "Kernel Size (must be odd)",
        "Pixel Intensity Threshold",
        "Distance Threshold",
    ],
    "Non-Local Means Denoising": ["Strentgh", "Template Window Size", "search window size"],
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
cv2.createTrackbar(denoise_trackbar_0_name, denoise_window_name, 1, 51, cb_denoise_trackbar_0)
cv2.createTrackbar(denoise_trackbar_1_name, denoise_window_name, 1, 300, cb_denoise_trackbar_1)
cv2.createTrackbar(denoise_trackbar_2_name, denoise_window_name, 1, 300, cb_denoise_trackbar_2)

image_idx = cb_image_trackbar(0)
denoise_algorithm_idx = cb_denoise_algorithm_trackbar(0)
denoise_params = [cb_denoise_trackbar_0(1), cb_denoise_trackbar_1(1), cb_denoise_trackbar_2(1)]

cb_edges_algorithm_trackbar = lambda val: val
cb_edges_trackbar_0 = lambda val: val
cb_edges_trackbar_1 = lambda val: val
cb_edges_trackbar_2 = lambda val: val

edges_algorithm_trackbar_name = "Edge Detection Alg. #"
edges_trackbar_0_name = "Param 1"
edges_trackbar_1_name = "Param 2"
edges_trackbar_2_name = "Param 3"
edges_window_name = "Detected Edges"
edges_algorithms = {
    "No Edge Detection": [],
    "Canny Edge Detection": ["Low Threshold", "High Threshold"],
    "Sobel Edge Detection": ["Kernel Size (must be odd)", "dx Order", "dy Order"],
    "Laplacian Edge Detection": ["Kernel Size (must be odd)"],
}
edges_algorithm_names = list(edges_algorithms.keys())

cv2.namedWindow(edges_window_name)
cv2.createTrackbar(
    edges_algorithm_trackbar_name,
    edges_window_name,
    0,
    len(edges_algorithms) - 1,
    cb_edges_algorithm_trackbar,
)
cv2.createTrackbar(edges_trackbar_0_name, edges_window_name, 1, 200, cb_edges_trackbar_0)
cv2.createTrackbar(edges_trackbar_1_name, edges_window_name, 1, 1000, cb_edges_trackbar_1)
cv2.createTrackbar(edges_trackbar_2_name, edges_window_name, 1, 1000, cb_edges_trackbar_2)

edges_algorithm_idx = cb_edges_algorithm_trackbar(0)
edges_param_1 = cb_edges_trackbar_0(1)
edges_param_2 = cb_edges_trackbar_1(1)
edges_param_3 = cb_edges_trackbar_2(1)

key = cv2.waitKey(1)
while key != 113:
    # Load image
    image_idx = cv2.getTrackbarPos(image_trackbar_name, denoise_window_name)

    # Get denoising parameters
    denoise_algorithm_idx = cv2.getTrackbarPos(denoise_algorithm_trackbar_name, denoise_window_name)
    denoise_params = [0, 0, 0]
    denoise_params[0] = cv2.getTrackbarPos(denoise_trackbar_0_name, denoise_window_name)
    denoise_params[1] = cv2.getTrackbarPos(denoise_trackbar_1_name, denoise_window_name)
    denoise_params[2] = cv2.getTrackbarPos(denoise_trackbar_2_name, denoise_window_name)

    denoise_algorithm_name = denoise_algorithm_names[denoise_algorithm_idx]
    denoise_param_names = denoise_algorithms[denoise_algorithm_name]

    image_path = os.path.join(images_dir, images[image_idx])
    image_full = cv2.imread(image_path)
    image = downscale_image(image_full)
    start = time.time()
    image_denoised = denoise_image(
        image,
        denoise_algorithm_name,
        *denoise_params,
    )
    end = time.time()
    denoised_time = end - start

    # Display denoised image
    image_denoised_drawn = draw_legend(
        image_denoised.copy(), denoise_algorithm_name, denoised_time, denoise_param_names
    )
    cv2.imshow(denoise_window_name, image_denoised_drawn)

    # Get edge detection parameters
    edges_algorithm_idx = cv2.getTrackbarPos(edges_algorithm_trackbar_name, edges_window_name)
    edges_params = [0, 0, 0]
    edges_params[0] = cv2.getTrackbarPos(edges_trackbar_0_name, edges_window_name)
    edges_params[1] = cv2.getTrackbarPos(edges_trackbar_1_name, edges_window_name)
    edges_params[2] = cv2.getTrackbarPos(edges_trackbar_2_name, edges_window_name)

    edges_algorithm_name = edges_algorithm_names[edges_algorithm_idx]
    edges_param_names = edges_algorithms[edges_algorithm_name]

    # Detect edges
    start = time.time()
    image_edges = detect_edges(image_denoised.copy(), edges_algorithm_name, *edges_params)
    end = time.time()
    edges_time = end - start

    # Display detected edges
    image_edges_drawn = draw_legend(
        image_edges.copy(), edges_algorithm_name, edges_time, edges_param_names
    )
    cv2.imshow(edges_window_name, image_edges_drawn)
    key = cv2.waitKey(1)

cv2.destroyAllWindows()
