# -*- coding: utf-8 -*-
"""Face processing

Augment the original dataset and save them to the local folder.

Usage: python imgAug.py -p <path_of_folder>
@auther:Qingkuo Li, Xueni Fu
"""
import argparse
import Augmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'imgAug',
                    description = 'Augment the original dataset and save them to the local folder.')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path of the input dataset folder')
    args = parser.parse_args()

    for i in range(0, 6):
        data_path = args.path + '/' + str(i) + '/'
        p = Augmentor.Pipeline(data_path)

        p.flip_left_right(probability=0.5)
        p.rotate(probability=0.1, max_left_rotation=3, max_right_rotation=3)
        p.shear(probability=0.1, max_shear_left=2, max_shear_right=2)
        p.skew(probability=0.05)
        p.random_distortion(probability=0.05, grid_width=1, grid_height=1, magnitude=1)
        p.random_brightness(probability=0.1, min_factor=0.9, max_factor=1.1)
        p.random_contrast(probability=0.1, min_factor=0.9, max_factor=1.1)

        num_of_samples = 5000
        p.sample(num_of_samples)