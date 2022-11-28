# -*- coding: utf-8 -*-
"""Convert csv to images

Convert csv file to images and save them accoding to labels.
Only used for testing if img2csv module works normally.
This script does not play any role in the training and processing.

Usage: python csv2img.py -f <csv_file> -p <output_path>
@author: Qingkuo Li, Xueni Fu
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image

# 0-angry, 1-fear, 2-happy, 3-sad, 4-surprise, 6-neutral
def csv_to_img(csv_path, output_path, verbose=False):
    for save_path, csv_file in [(output_path, csv_path)]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num = 1
        with open(csv_file) as f:
            csvr = csv.reader(f)
            header = next(csvr) # jump header
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                img = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                if verbose:
                    print(image_name)
                img.save(image_name)
                num  = num + 1

    print('Convering is completed (total: {})'.format(num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog = 'csv2img',
                description = 'Convert csv file to images and save them accoding to labels.')
    parser.add_argument('-f', '--file', required=True, type=str, help='csv file')
    parser.add_argument('-p', '--path', required=True, type=str, help='output path')
    parser.add_argument('-v', '--verbose', action='store_true', help="vebose")

    args = parser.parse_args()

    csv_path = args.file
    output_path = args.path
    csv_to_img(csv_path, output_path, args.verbose)