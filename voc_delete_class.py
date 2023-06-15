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
    description="Script for deleting all instances of a particular class name from a Pascal VOC formatted dataset file"
)
parser.add_argument(
    "--input", "-i", type=str, required=True, help="Path to the dataset file to be changed"
)
parser.add_argument(
    "--output", "-o", type=str, required=False, help="Path to save the new dataset file"
)
parser.add_arugment(
    "--classname", "-n", required=True, help="Class name to be deleted from the dataset file"
)

arguments = parser.parse_args()
input_path = arguments.input
class_name = arguments.classname
if arguments.output:
    output_path = arguments.output
else:
    output_path = input_path

logger.info(f"======== Delete VOC Class ========")
logger.info(f"Input File Path: {input_path}")
logger.info(f"Output File Path: {output_path}")
logger.info(f"Class name: {class_name}")

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
    deleted = False
    if type == class_name:
        root.delete(label)
        deleted = True

    logger.debug(f"Type: {type}")
    logger.debug(f"Deleted: {deleted}")


try:
    tree.write(output_path)
except Exception as e:
    logger.error(f"Unable to write to {output_path}")
    logger.error(e)
    sys.exit(1)
