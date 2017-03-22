# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import csv
import cv2
import glob
import numpy as np
from functools import reduce

def process_coords(coords_file, delimiter=" "):
  parsed_coords = None
  resolved_file = os.path.join(os.getcwd(), coords_file)

  # tell the user that the path specified does not exist.
  if os.path.exists(resolved_file) is False:
    sys.stdout.write(f"Path {resolved_file} does not exist\n")
    sys.exit(0)
    return

  def reduce_fn(accumulated_value, value):
    # remove the empty strings and convert all items in the list to integer
    filtered_row = list(map(int, filter(None, value)))

    # chunk the list to smaller chunks consisting (x and y positions)
    chunk_coordinates = [filtered_row[i:i + 2] for i in range(0, len(filtered_row), 2)]

    # add to the list of accumulated values
    accumulated_value.append(chunk_coordinates)

    return accumulated_value

  with open(coords_file, "r") as csv_data_file:
    reader = csv.reader(csv_data_file, delimiter=delimiter)

    # reduce the the coordinates in to a single list of lists
    parsed_coords = reduce(reduce_fn, reader, [])

    # remove empty lists in the reduced_value
    parsed_coords = list(filter(None, parsed_coords))

  return parsed_coords

def create_cv_im_instance(image_path):
  return {
    'path': image_path,
    'cv_im': cv2.imread(image_path)
  }

def create_cv_im_instances_from_dir(image_dir_path, **kwargs):
  file_exts = kwargs.get('file_exts', ['jpg', 'png'])

  if not os.path.exists(image_dir_path):
    sys.stdout.write(f"Directory of Images: {image_dir_path} does not exist\n")
    sys.exit(1)

  # get all images in the directory that matches the extensions provided
  images = []
  for ext in file_exts:
    images.extend(glob.glob(f"{image_dir_path}/*.{ext}"))

  # terminate if no images are found
  if len(images) == 0:
    sys.stdout.write(f"No images in the source directory {image_dir_path}\n")
    sys.exit(1)

  # convert images list to cv image instances list
  return list(map(create_cv_im_instance, images))

def create_binary_image(cv_im):
  # convert to grayscale
  gray_image = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)

  _, processed = cv2.threshold(gray_image, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  kernel_opening = np.ones((5, 5), np.uint8)
  processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_opening)

  return processed


