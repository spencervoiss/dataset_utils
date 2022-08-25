# Lint as: python3
"""
FILE: splt_dataset.py
AUTH: Spencer Voiss
DATE: 25 Aug 2022
DESC: Split up a collection of individual datapoint 
  files into train, validation, and test datasets
"""

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
parser.add_argument("--output_dir", type=str, required=True,
  help="Directory for the split up dataset")
parser.add_argument('-x', "--extension", type=str,
  help="File extension to be considered (e.g. '.json'). If none\
    specified, will consider all file types in the input directory")
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
parser.add_argument("--loglevel", type=str, default="error",
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
output_dir = args.output_dir
test_split = args.test_split
train_split = args.train_split
valid_split = args.validation_split
try:
  assert (test_split + train_split + valid_split) == 1.0, "Dataset split must total 1"
except AssertionError as err:
  logging.error(err)

# Create the output directories
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
valid_dir = os.path.join(output_dir, "valid")
try:
  os.mkdir(output_dir)
except FileExistsError as err:
  logger.warning("Directory %s already exists" % output_dir)
try:
  os.mkdir(train_dir)
except FileExistsError as err:
  logger.warning("Train directory already exists. Files may be overwritten")
try:
  os.mkdir(valid_dir)
except FileExistsError as err:
  logger.warning("Validation directory already exists. Files may be overwritten")
try:
  os.mkdir(test_dir)
except FileExistsError as err:
  logger.warning("Test directory already exists. Files may be overwritten")

# Read in all files of the specified type (or all if none specified)
input_files = []
if args.extension:
  for file in os.listdir(input_dir):
    if file.endsswith(args.extension):
      input_files.append(file)
else:
  for file in os.listdir(input_dir):
    input_files.append(file)

# Randomize the order of the files to prevent overrepresentation
shuffle(input_files)

# Split the input files into their respective datasets
num_train_files = int(len(input_files)*train_split)
num_valid_files = int(len(input_files)*valid_split)
num_test_files = len(input_files) - num_train_files - num_valid_files 
train_files = []
valid_files = []
test_files = []
for idx in range(len(input_files)):
  if idx < num_train_files:
    train_files.append(input_files[idx])
  elif idx < (num_train_files + num_valid_files):
    valid_files.append(input_files[idx])
  elif idx < (num_train_files + num_valid_files + num_test_files):
    test_files.append(input_files[idx])

# Check to make sure all files will be split
len_input_files = len(input_files)
len_train_files = len(train_files)
len_valid_files = len(valid_files)
len_test_files = len(test_files)
hanging_files = len_input_files - len_train_files - len_valid_files - len_test_files

logger.debug("Dataset split:")
logger.debug("\tNumber of input files: %i"%len_input_files)
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
  shutil.copy(file_path, train_dir)
logger.info("Copying %i files into valid dir..." % len_valid_files)
for file in valid_files:
  file_path = os.path.join(input_dir, file)
  shutil.copy(file_path, valid_dir)
logger.info("Copying %i files into test dir..." % len_test_files)
for file in test_files:
  file_path = os.path.join(input_dir, file)
  shutil.copy(file_path, test_dir)
logger.info("All files copied! Your dataset is now split")