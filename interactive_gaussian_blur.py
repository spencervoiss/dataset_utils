import argparse
import os
import time

import cv2

global image
global downscaled_image
global images
global kernel_size
kernel_size = 5
global sigma
sigma = 0
WINDOW_TITLE = "Gaussian Blur"


def image_trackbar(val):
    image = cv2.imread(os.path.join(imgs_dir, images[val - 1]))
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2
    global downscaled_image
    global kernel_size
    global sigma
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(downscaled_image, (kernel_size, kernel_size), sigma)
    cv2.imshow(WINDOW_TITLE, blurred)


def kernel_trackbar(val):
    global kernel_size
    kernel_size = val
    global downscaled_image
    global sigma
    blurred = cv2.GaussianBlur(downscaled_image, (kernel_size, kernel_size), sigma)
    cv2.imshow(WINDOW_TITLE, blurred)


def sigma_trackbar(val):
    global kernel_size
    global downscaled_image
    global sigma
    sigma = val
    blurred = cv2.GaussianBlur(downscaled_image, (kernel_size, kernel_size), sigma)
    cv2.imshow(WINDOW_TITLE, blurred)


parser = argparse.ArgumentParser(description="Interactive gaussian blur visualizer")
parser.add_argument(
    "-i", "--input_dir", help="Path to directory containing the images to be blurred"
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
sigma_trackbar_name = "Sigma"
cv2.createTrackbar(sigma_trackbar_name, WINDOW_TITLE, 0, 5, sigma_trackbar)
image_trackbar(0)
kernel_trackbar(kernel_size)
sigma_trackbar(sigma)
cv2.waitKey()
