import argparse
import os
import time

import cv2

global image
global downscaled_image
global images
global kernel_size
kernel_size = 5
global pixel_intensity_threshold
pixel_intensity_threshold = 75
global spatial_distance_threshold
spatial_distance_threshold = 75
WINDOW_TITLE = "Bilateral Filter"


def image_trackbar(val):
    image = cv2.imread(os.path.join(imgs_dir, images[val - 1]))
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2
    global downscaled_image
    global kernel_size
    global pixel_intensity_threshold
    global spatial_distance_threshold
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    filtered = cv2.bilateralFilter(
        downscaled_image, kernel_size, pixel_intensity_threshold, spatial_distance_threshold
    )
    cv2.imshow(WINDOW_TITLE, filtered)


def kernel_trackbar(val):
    global kernel_size
    kernel_size = val
    global downscaled_image
    global pixel_intensity_threshold
    global spatial_distance_threshold
    filtered = cv2.bilateralFilter(
        downscaled_image, kernel_size, pixel_intensity_threshold, spatial_distance_threshold
    )
    cv2.imshow(WINDOW_TITLE, filtered)


def pixel_intensity_threshold_trackbar(val):
    global kernel_size
    global downscaled_image
    global pixel_intensity_threshold
    global spatial_distance_threshold
    pixel_intensity_threshold = val
    filtered = cv2.bilateralFilter(
        downscaled_image, kernel_size, pixel_intensity_threshold, spatial_distance_threshold
    )
    cv2.imshow(WINDOW_TITLE, filtered)


def spatial_distance_threshold_trackbar(val):
    global kernel_size
    global downscaled_image
    global pixel_intensity_threshold
    global spatial_distance_threshold
    spatial_distance_threshold = val
    filtered = cv2.bilateralFilter(
        downscaled_image, kernel_size, pixel_intensity_threshold, spatial_distance_threshold
    )
    cv2.imshow(WINDOW_TITLE, filtered)


parser = argparse.ArgumentParser(description="Interactive bilateral filter visualizer")
parser.add_argument(
    "-i", "--input_dir", help="Path to directory containing the images to be filtered"
)
args = parser.parse_args()
imgs_dir = args.input_dir

try:
    images = os.listdir(imgs_dir)
except:
    print("Input directory is empty")
    exit(1)
cv2.namedWindow(WINDOW_TITLE)
img_trackbar_name = f"Image"
cv2.createTrackbar(img_trackbar_name, WINDOW_TITLE, 0, len(images), image_trackbar)
kernel_trackbar_name = "Kernel Size"
cv2.createTrackbar(kernel_trackbar_name, WINDOW_TITLE, 1, 21, kernel_trackbar)
pixel_intensity_threshold_trackbar_name = "pixel intensity threshold"
cv2.createTrackbar(
    pixel_intensity_threshold_trackbar_name,
    WINDOW_TITLE,
    1,
    1000,
    pixel_intensity_threshold_trackbar,
)
spatial_distance_threshold_trackbar_name = "Spatial Distance Threshold"
cv2.createTrackbar(
    spatial_distance_threshold_trackbar_name,
    WINDOW_TITLE,
    1,
    15,
    spatial_distance_threshold_trackbar,
)
image_trackbar(0)
kernel_trackbar(kernel_size)
pixel_intensity_threshold_trackbar(pixel_intensity_threshold)
spatial_distance_threshold_trackbar(spatial_distance_threshold)
cv2.waitKey()
