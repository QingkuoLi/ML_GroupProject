# -*- coding: utf-8 -*-
"""Convert images to csv

Convert images (in labeled sub-folders) to a single csv-format file.

Usage: python img2csv.py -p <data_folder>
@auther:Qingkuo Li,Xueni Fu
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'img2csv',
                    description = 'Convert images (in labeled sub-folders) to a single csv-format file.')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path of the input dataset folder')
    args = parser.parse_args()
    input_path = args.path
    output_file = input_path + '/dataset.csv'

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'pixels'])

        for rootdir, dirs, files in os.walk(input_path):
            for subdir in dirs:
                if os.path.isdir(os.path.join(rootdir, subdir)):
                    for file in os.listdir(os.path.join(rootdir, subdir)):
                        file_name = os.path.join(os.path.join(rootdir, subdir), file)
                        img_gray = Image.open(file_name)
                        img_gray = np.array(img_gray.getdata(), dtype=np.int)
                        img_gray = img_gray.flatten()
                        img_gray = ' '.join(map(str, img_gray))
                        writer.writerow([subdir, img_gray])