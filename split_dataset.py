# Lint as: python3
"""
FILE: split_dataset.py
AUTH: Spencer Voiss
DATE: 25 Aug 2022
DESC: Split up a collection of individual datapoint 
  files into train, validation, and test datasets
"""

from __future__ import annotations
from argparse import ArgumentParser
from operator import truediv
import os
import sys
import logging
from random import shuffle
import shutil

parser = ArgumentParser(description = 
  'Split your dataset into train, validation, and test sets')

parser.add_argument("--input_dir", type=str, required=True,
  help="Directory containing the files to be split up")
parser.add_argument("--img_dir", type=str, required=False,
  help="Directory containing images corresponding to the inputs")
parser.add_argument("--output_dir", type=str, required=True,
  help="Directory for the split up dataset")
parser.add_argument('-x', "--extension", type=str, required=True,
  help="File extension to be considered (e.g. '.json').") 
parser.add_argument('--img_extension', type=str, required="--img_dir" in sys.argv,
  help="File extension type for the images to be considered")
parser.add_argument("-t", "--train_split", type=float,
  default=0.7,
  help="Percentage of files to go to the train dataset")
parser.add_argument("-v", "--validation_split", type=float,
  default=0.2,
  help="Percentage of files to go to the validation dataset")
parser.add_argument("-e", "--test_split", type=float, 
  default=0.1,
  help="Percentage of files to go to the test dataset")
parser.add_argument("--logfile", type=str, default="/dev/null",
  help="File to save the log to. If not specified, log will not be saved")
parser.add_argument("--loglevel", type=str, default="info",
  help="Level to log at. Acceptable arguments are debug, info, warning, error, and critical")

args = parser.parse_args()
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
  handlers=[
    logging.FileHandler(args.logfile),
    logging.StreamHandler(sys.stdout)
  ] 
  )
logging.info("Writing log to %(args.logfile)s")
logger = logging.getLogger()
try:
  assert args.loglevel in ["cricital", "error", "warning", "info", "debug"], "Logging level must be debug, info, warning, error, or critical. Setting logging level to INFO..."
except AssertionError as err:
  logger.setLevel(logging.INFO)
  logging.warning(err)
if args.loglevel == "error":
  logger.setLevel(logging.ERROR)
elif args.loglevel == "warning":
  logger.setLevel(logging.WARNING)
elif args.loglevel == "info":
  logger.setLevel(logging.INFO)
elif args.loglevel == "debug":
  logger.setLevel(logging.DEBUG)

# Parse arguments
input_dir = args.input_dir
input_extension = args.extension
output_dir = args.output_dir
img_dir = args.img_dir
img_extension = args.img_extension
test_split = args.test_split
train_split = args.train_split
valid_split = args.validation_split
try:
  assert (test_split + train_split + valid_split) == 1.0, "Dataset split must total 1"
except AssertionError as err:
  logging.error(err)

# Create the output directories
train_data_dir = os.path.join(output_dir, "annotations/train")
test_data_dir = os.path.join(output_dir, "annotations/test")
valid_data_dir = os.path.join(output_dir, "annotations/valid")
train_imgs_dir = os.path.join(output_dir, "images/train")
valid_imgs_dir = os.path.join(output_dir, "images/valid")
test_imgs_dir = os.path.join(output_dir, "images/test")
annotations_dir = os.path.join(output_dir, "annotations")
images_dir = os.path.join(output_dir, "images")

try:
  os.mkdir(output_dir)
except FileExistsError as err:
  logger.warning("Directory %s already exists" % output_dir)
try:
  os.mkdir(annotations_dir)
except FileExistsError as err:
  logger.warning("Annotations directory already exists. Files may be overwritten")
try:
  os.mkdir(images_dir)
except FileExistsError as err:
  logger.warning("Images directory already exists. Files may be overwritten")
try:
  os.mkdir(train_data_dir)
except FileExistsError as err:
  logger.warning("Train directory already exists. Files may be overwritten")
try:
  os.mkdir(valid_data_dir)
except FileExistsError as err:
  logger.warning("Validation directory already exists. Files may be overwritten")
try:
  os.mkdir(test_data_dir)
except FileExistsError as err:
  logger.warning("Test directory already exists. Files may be overwritten")
if img_dir:
  try:
    os.mkdir(test_imgs_dir)
  except FileExistsError as err:
    logger.warning("Test images directory already exists. Files may be overwritten")
  try:
    os.mkdir(train_imgs_dir)
  except FileExistsError as err:
    logger.warning("Train images directory already exists. Files may be overwritten")
  try:
    os.mkdir(valid_imgs_dir)
  except FileExistsError as err:
    logger.warning("Valid images directory already exists. Files may be overwritten")

# Read in all files of the specified type 
inputs_noExtension = []
for file in os.listdir(input_dir):
  filename, ext = os.path.splitext(file)
  if ext == input_extension:
    inputs_noExtension.append(filename)

# Randomize the order of the files to prevent overrepresentation
shuffle(inputs_noExtension)

# Split the input files into their respective datasets
num_train_files = int(len(inputs_noExtension)*train_split)
num_valid_files = int(len(inputs_noExtension)*valid_split)
num_test_files = len(inputs_noExtension) - num_train_files - num_valid_files 
train_files = []
valid_files = []
test_files = []
train_imgs = []
valid_imgs = []
test_imgs = []
for idx in range(len(inputs_noExtension)):
  filename = inputs_noExtension[idx] + input_extension
  if idx < num_train_files:
    train_files.append(filename)
  elif idx < (num_train_files + num_valid_files):
    valid_files.append(filename)
  elif idx < (num_train_files + num_valid_files + num_test_files):
    test_files.append(filename)
if img_dir:
  for idx in range(len(inputs_noExtension)):
    filename = inputs_noExtension[idx] + img_extension
    if idx < num_train_files:
      train_imgs.append(filename)
    elif idx < (num_train_files + num_valid_files):
      valid_imgs.append(filename)
    elif idx < (num_train_files + num_valid_files + num_test_files):
      test_imgs.append(filename)

# Check to make sure all files will be split
len_inputs_noExtension = len(inputs_noExtension)
len_train_files = len(train_files)
len_valid_files = len(valid_files)
len_test_files = len(test_files)
hanging_files = len_inputs_noExtension - len_train_files - len_valid_files - len_test_files

logger.debug("Dataset split:")
logger.debug("\tNumber of input files: %i"%len_inputs_noExtension)
logger.debug("\tNumber of training files: %i"%len_train_files)
logger.debug("\tNumber of validation files: %i"%len_valid_files)
logger.debug("\tNumber of testing files: %i"%len_test_files)
try:
  assert hanging_files == 0, "%i files haven't been split"%hanging_files
except AssertionError as err:
  logger.exception(msg=err)

# Copy files to their respective directories
logger.info("Copying %i files into train dir..." % len_train_files)
for file in train_files:
  file_path = os.path.join(input_dir, file)
  shutil.copy(file_path, train_data_dir)
logger.info("Copying %i files into valid dir..." % len_valid_files)
for file in valid_files:
  file_path = os.path.join(input_dir, file)
  shutil.copy(file_path, valid_data_dir)
logger.info("Copying %i files into test dir..." % len_test_files)
for file in test_files:
  file_path = os.path.join(input_dir, file)
  shutil.copy(file_path, test_data_dir)
if img_dir:
  logger.info("Copying %i files into train images dir..." % len_train_files)
  for file in train_imgs:
    file_path = os.path.join(img_dir, file)
    shutil.copy(file_path, train_imgs_dir)
  logger.info("Copying %i files into validation images dir..." % len_valid_files)
  for file in valid_imgs:
    file_path = os.path.join(img_dir, file)
    shutil.copy(file_path, valid_imgs_dir)
  logger.info("Copying %i files into test images dir..." % len_test_files)
  for file in test_imgs:
    file_path = os.path.join(img_dir, file)
    shutil.copy(file_path, test_imgs_dir)
logger.info("All files copied! Your dataset is now split")
