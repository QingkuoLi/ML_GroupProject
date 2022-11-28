# -*- coding: utf-8 -*-
"""Face processing

Resize & convert images to gray-scale and save to local folder.

Usage: python faceprocess.py -p <path_of_folder>
@auther:Qingkuo Li,Xueni Fu
"""
import os
import cv2
import argparse
from PIL import Image
from os.path import join

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'faceprocess',
                    description = 'Crop face-ROI and save to local folder.')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path of the input folder')
    parser.add_argument('-v', '--verbose', action='store_true', help="vebose")
    args = parser.parse_args()
    input_path = args.path

    count = 1
    for rootdir, dirs, files in os.walk(input_path):
        for subdir in dirs:
            if os.path.isdir(os.path.join(rootdir, subdir)):
                output_path = rootdir + '/processed/' + subdir + '/'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for file in os.listdir(os.path.join(rootdir, subdir)):
                    file_name = os.path.join(os.path.join(rootdir, subdir), file)
                    img_gray = Image.open(file_name).convert('L')
                    img_gray = img_gray.resize((48, 48), Image.Resampling.LANCZOS)
                    output_file_name = output_path + str(count) + '.jpg'
                    if args.verbose:
                        print('Processing file: {}'.format(output_file_name))
                    img_gray.save(output_file_name)
                    count = count + 1

    print('Face image processing is completed (total: {})'.format(count))