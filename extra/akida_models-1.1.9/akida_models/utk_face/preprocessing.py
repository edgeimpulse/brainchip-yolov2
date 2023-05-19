#!/usr/bin/env python
# ******************************************************************************
# Copyright 2020 Brainchip Holdings Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Toolset to load UTKFace preprocessed dataset.

Original files: https://susanqq.github.io/UTKFace/

"""

import os
import numpy as np
import tensorflow as tf

from ..utils import fetch_file


def _load_images(root_path):
    """ Load images files and labels from the disk.

    Args:
        root_path (str): path containing image files

    Returns:
        a set of images and their associated labels as a tuple of numpy arrays
    """
    file_list = os.listdir(root_path)
    num_files = len(file_list)

    x = np.zeros((num_files, 32, 32, 3))
    for i in range(num_files):
        raw_image = tf.io.read_file(os.path.join(root_path, file_list[i]))
        image = tf.image.decode_jpeg(raw_image, channels=3)
        x[i, :, :, :] = np.expand_dims(image.numpy(), axis=0)

    y = np.array([[i.split('_')[0]] for i in file_list]).astype('int8')

    return x, y


def load_data():
    """ Loads the dataset from Brainchip data server.

    Returns:
        np.array, np.array, np.array, np.array: train set, train labels, test
        set and test labels as numpy arrays

    """
    dataset = fetch_file(
        "http://data.brainchip.com/dataset-mirror/utk_face/UTKFace_preprocessed.tar.gz",
        fname="UTKFace_preprocessed.tar.gz",
        cache_subdir='datasets/',
        extract=True)
    file_path = os.path.join(os.path.dirname(dataset), "UTKFace")

    train_path = os.path.join(file_path, "train")
    test_path = os.path.join(file_path, "test")

    x_train, y_train = _load_images(train_path)
    x_test, y_test = _load_images(test_path)

    return x_train, y_train, x_test, y_test
