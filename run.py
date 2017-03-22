#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import click
import numpy as np
from app import process_coords, create_binary_image, create_cv_im_instances_from_dir

@click.command()
@click.argument('images-dir', type=click.Path(exists=True))
@click.argument('output-dir', type=click.Path())
@click.option('--show-logs/--no-show-logs', default=True, help='Prints out the information on what file is processed and other information.')
@click.option('--extension', default=['jpg'], multiple=True, help='Image file extensions to match (e.g. jpg).')
@click.option('--coords-file', default='assets/docs/fields39.csv', type=click.Path(exists=True), help='Image file extensions to match (e.g. jpg).')
def processor(images_dir, output_dir, show_logs, extension, coords_file):
  parsed_coords = process_coords(os.path.join(os.getcwd(), coords_file))

  # resolve the destination folder
  destination_folder = os.path.join(os.getcwd(), output_dir)

  # read the images from the directory
  cv_images = create_cv_im_instances_from_dir(images_dir)

  if show_logs is True:
    click.echo(f"Reading images from the directory: {images_dir}")
    click.echo(f"Saving processed files to: {destination_folder}")

  # create the folder if it does not exist
  if os.path.exists(destination_folder) is False:
    os.mkdir(destination_folder)

  for item in cv_images:
    img_basename = os.path.basename(item['path'])
    cv_image = item['cv_im']

    if show_logs is True:
      click.echo(f"Processing Image: {item['path']}")

    # conver the image to binary
    processed = create_binary_image(cv_image)

    shape_thickness = 2
    crossed_shape_color = (255, 0, 0)
    shaded_shape_color = (0, 0, 255)
    blank_shape_color = (0, 255, 0)

    radius = 8

    statistics = {
      'SA': 0,
      'A': 0,
      'SLA': 0,
      'NAD': 0,
      'SLD': 0,
      'D': 0,
      'SD': 0
    }

    statistics_keys = list(statistics.keys())

    for row in parsed_coords:
      for idx, coords in enumerate(row):
        # flag to determine if the item is shaded or not
        is_shaded = False

        # identify where the item resides in the choices
        stat_key = statistics_keys[idx]

        # extract x and y coordinates
        [x, y] = coords

        # get corners of the border based on the x and y coords along with the radius
        top = y - radius
        left = x - radius
        right = x + radius
        bottom = y + radius

        # crop the image to the specific region by the coordinates of top, right, bottom, left
        # or also known as region of interest
        roi = processed[top:bottom, left:right]

        # opencv's mean function processes all 4 channels of the image
        # and we need only the first channel since we are processing binary images
        # and discard all the channels other the first channel
        #
        # This to determine how much black pixels are present on the cropped area
        (channel1_mean, _, _, _) = cv2.mean(roi)

        # define default color for the rectangle
        shape_color = blank_shape_color

        # override the default rectangle color
        if channel1_mean < 69:
          shape_color = crossed_shape_color
        elif channel1_mean < 135 and channel1_mean > 69:
          shape_color = shaded_shape_color

          is_shaded = True

        # draw the rectangles
        cv2.rectangle(cv_image, (left, top), (right, bottom), shape_color, 2)

        # add one to the matched stat value
        if is_shaded is True:
          statistics[stat_key] += 1

    # write the manipulated image to the detsination folder
    cv2.imwrite(f"{destination_folder}/{img_basename}", cv_image)

    # save the statistics in a text file
    with open(f"{destination_folder}/{img_basename}.txt", 'w') as stat_file:
      for key, val in statistics.items():
        stat_file.write(f"{key} = {val}\n")

  if show_logs is True:
    click.echo(f"Processing images done. Output files on: {destination_folder}")

if __name__ == '__main__':
  processor()


