"""Swaps the x and y coordinates of a Pascal VOC xml formatted dataset file
Arguments:
path -- Path to the dataset file to be converted
output -- Path to save the coordinate swapped dataset file
"""

import argparse
import logging
import sys
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s", filemode="a"
)

logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description="Script for changing all floating point values to integers in a Pascal VOC dataset file"
)
parser.add_argument(
    "--input", "-i", type=str, required=True, help="Path to the dataset file to be changed"
)
parser.add_argument(
    "--output", "-o", type=str, required=False, help="Path to save the new dataset file"
)

arguments = parser.parse_args()
input_path = arguments.input
if arguments.output:
    output_path = arguments.output
else:
    output_path = input_path

logger.info(f"======== Swap VOC Coordinates ========")
logger.info(f"Input File Path: {input_path}")
logger.info(f"Output File Path: {output_path}")

tree = ET
try:
    tree = ET.parse(input_path)
except Exception as e:
    logger.fatal(f"Unable to read {input_path}. File is not xml or does not exist")
    logger.error(e)
    sys.exit(1)

root = tree.getroot()
for label in root.findall("object"):
    bbox = label.find("bndbox")
    type = label.find("name").text
    xmin_attr = bbox.find("xmin")
    ymin_attr = bbox.find("ymin")
    xmax_attr = bbox.find("xmax")
    ymax_attr = bbox.find("ymax")

    xmin = xmin_attr.text
    ymin = ymin_attr.text
    xmax = xmax_attr.text
    ymax = ymax_attr.text

    logger.debug(f"Type: {type}")
    logger.debug(f"Original Values:")
    logger.debug(f"\txmin: {xmin}")
    logger.debug(f"\tymin: {ymin}")
    logger.debug(f"\txmax: {xmax}")
    logger.debug(f"\tymax: {ymax}")

    xmin_new = str(int(float(xmin)))
    ymin_new = str(int(float(ymin)))
    xmax_new = str(int(float(xmax)))
    ymax_new = str(int(float(ymax)))
    logger.debug(f"Swapped Values")
    logger.debug(f"\txmin: {xmin_new}")
    logger.debug(f"\tymin: {ymin_new}")
    logger.debug(f"\txmax: {xmax_new}")
    logger.debug(f"\tymax: {ymax_new}")

    xmin_attr.text = xmin_new
    ymin_attr.text = ymin_new
    xmax_attr.text = xmax_new
    ymax_attr.text = ymax_new

    logger.debug(f"New ojbect")
    logger.debug(f"\txmin: {bbox.find('xmin').text}")
    logger.debug(f"\tymin: {bbox.find('ymin').text}")
    logger.debug(f"\txmax: {bbox.find('xmax').text}")
    logger.debug(f"\tymax: {bbox.find('ymax').text}")

logger.debug(f"Swapped Values")
for label in root.findall("object"):
    bbox = label.find("bndbox")
    type = label.find("name").text
    xmin_attr = bbox.find("xmin")
    ymin_attr = bbox.find("ymin")
    xmax_attr = bbox.find("xmax")
    ymax_attr = bbox.find("ymax")

    xmin = xmin_attr.text
    ymin = ymin_attr.text
    xmax = xmax_attr.text
    ymax = ymax_attr.text

    logger.debug(f"Type: {type}")
    logger.debug(f"\txmin: {xmin}")
    logger.debug(f"\tymin: {ymin}")
    logger.debug(f"\txmax: {xmax}")
    logger.debug(f"\tymax: {ymax}")

try:
    tree.write(output_path)
except Exception as e:
    logger.error(f"Unable to write to {output_path}")
    logger.error(e)
    sys.exit(1)
