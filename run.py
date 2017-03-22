#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import csv
import cv2
import numpy as np
from functools import reduce

###################################
# [parsing coordinates file] ::start
###################################

parsed_coords = None
data_file = os.path.join(os.getcwd(), "assets/docs/fields39.csv")
with open(data_file, "r") as csv_data_file:
  reader = csv.reader(csv_data_file, delimiter=" ")

  def reduce_fn(accumulated_value, value):
    # remove the empty strings and convert all items in the list to integer
    filtered_row = list(map(int, filter(None, value)))

    # chunk the list to smaller chunks consisting (x and y positions)
    chunk_coordinates = [filtered_row[i:i + 2] for i in range(0, len(filtered_row), 2)]

    # add to the list of accumulated values
    accumulated_value.append(chunk_coordinates)

    return accumulated_value

  # reduce the the coordinates in to a single list of lists
  parsed_coords = reduce(reduce_fn, reader, [])

  # remove empty lists in the reduced_value
  parsed_coords = list(filter(None, parsed_coords))

###################################
# [parsing coordinates file] ::end
###################################

###################################
# [image processing] ::start
###################################

# read the image
image = os.path.join(os.getcwd(), "assets/img/forms/0002.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

_, processed = cv2.threshold(gray_image ,0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel_opening = np.ones((5, 5), np.uint8)
processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_opening)

shape_thickness = 2
crossed_shape_color = (255, 0, 0)
shaded_shape_color = (0, 0, 255)
blank_shape_color = (0, 255, 0)

radius = 8

for row in parsed_coords:
  for coords in row:
    # extract x and y coordinates
    [x, y] = coords

    # get corners of the border based on the x and y coords along with the radius
    top = y - radius
    left = x - radius
    right = x + radius
    bottom = y + radius

    # region of interest
    roi = processed[top:bottom, left:right]

    # opencv's mean function processes all 4 channels of the image
    # and we need only the first channel since we are processing binary images
    # and discard all the channels other the first channel
    (channel1_mean, _, _, _) = cv2.mean(roi)

    # define default color for the rectangle
    shape_color = blank_shape_color

    # override the default rectangle color
    if channel1_mean < 69:
      shape_color = crossed_shape_color
    elif channel1_mean < 135 and channel1_mean > 69:
      shape_color = shaded_shape_color

    # draw the rectangles
    cv2.rectangle(cv_image, (left, top), (right, bottom), shape_color, 2)

cv2.imshow('test', cv2.pyrDown(cv_image))
cv2.waitKey(0)

###################################
# [image processing] ::end
###################################


