# ******************************************************************************
# Copyright 2021 Brainchip Holdings Ltd.
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
Preprocessing tools for ModelNet40 data handling.
"""

import os
import glob
import numpy as np
import tensorflow as tf
import trimesh

from .pointnet_utils import pointnet_preproc
from ..utils import fetch_file


def parse_dataset(data_dir, num_points=1024):
    """ Parse the dataset.

    Args:
        data_dir (str): directory that contains the dataset.
        num_points (int, optional): number of points  with which mesh is sample.
            Defaults to 1024.

    Returns:
        tuple: tuple of train and test points with the corresponding labels.
    """
    print('Preparing Dataset. May take a few minutes')
    print('This is only done once for a given requested number of points')
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_dir, "*"))
    folders.sort()

    i = -1
    for folder in folders:
        if not folder.endswith('README.txt'):
            i += 1
            print("processing class: {}".format(os.path.basename(folder)))
            # store folder name with ID so we can retrieve later
            class_map[i] = os.path.basename(os.path.normpath(folder))
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))

            # Raw files have extremely varying amplitude ranges
            # Here, we normalise all to the range [-1, 1]
            for f in train_files:
                temp_points = np.array(trimesh.load(f).sample(num_points))
                temp_points /= np.amax(np.abs(temp_points))
                train_points.append(temp_points)
                train_labels.append(i)

            for f in test_files:
                temp_points = np.array(trimesh.load(f).sample(num_points))
                temp_points /= np.amax(np.abs(temp_points))
                test_points.append(temp_points)
                test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def data_transformation(points, label):
    """ Data transformation function.

    Args:
        points (tf.Tensor): the points to which jitter is added.
        label (tf.Tensor): the labels.

    Returns:
        tf.Tensor, tf.Tensor: the points and labels for data transformation.
    """
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def get_modelnet_from_file(num_points, filename="ModelNet40.zip"):
    """ Load the ModelNet data from file.

    First parse through the ModelNet data folders. Each mesh is loaded and
    sampled into a point cloud before being added to a standard python list and
    converted to a `numpy` array. We also store the current enumerate index
    value as the object label and use a dictionary to recall this later.

    Args:
        num_points (int): number of points with which mesh is sample.
        filename (str): the dataset file to load if the npz file was not
            generated yet. Defaults to "ModelNet40.zip".

    Returns:
        np.array, np.array, np.array, np.array, dict: train set, train labels,
        test set, test labels as numpy arrays and dict containing class
        folder name.
    """

    base = os.path.basename(filename)
    short_filename = os.path.splitext(base)[0]

    datafile = short_filename + "_" + str(num_points) + "pts.npz"
    if os.path.exists(datafile):
        loadin = np.load(datafile, allow_pickle=True)
        train_points = loadin['train_points']
        train_labels = loadin['train_labels']
        test_points = loadin['test_points']
        test_labels = loadin['test_labels']
        class_map = loadin['class_map'].item()
    else:
        data_dir = filename

        # Join the test directory if needed
        is_exist = os.path.exists(data_dir)
        if not is_exist:
            data_dir = os.path.join("modelnet40", filename)
            is_exist = os.path.exists(data_dir)

        # Download dataset if not yet on local
        if not is_exist:
            original_url = "http://modelnet.cs.princeton.edu/" + filename
            # Load dataset
            # For ModelNet40, 2 Gb to download, and 12K files to unzip,
            # so can take some time
            print("Checking for downloaded, unzipped data (one-time task)...")
            data_dir = fetch_file(original_url, fname=filename, extract=True)
            data_dir = os.path.join(os.path.dirname(data_dir), short_filename)

        # Then read in the data and store a fixed number of points per sample
        train_points, test_points, train_labels, test_labels, class_map = parse_dataset(
            data_dir, num_points)
        np.savez(datafile,
                 train_points=train_points,
                 train_labels=train_labels,
                 test_points=test_points,
                 test_labels=test_labels,
                 class_map=class_map)
    return train_points, train_labels, test_points, test_labels, class_map


def get_modelnet(train_points,
                 train_labels,
                 test_points,
                 test_labels,
                 batch_size,
                 selected_points=128,
                 knn_points=64):
    """ Obtains the ModelNet dataset.

    Args:
        train_points (numpy.array): train set.
        train_labels (numpy.array): train labels.
        test_points (numpy.array): test set.
        test_labels (numpy.array): test labels.
        batch_size (int): size of the batch.
        selected_points (int): num points to process per sample. Default is 512.
        knn_points (int): number of points to include in each localised group.
            Must be a power of 2, and ideally an integer square (so 64, or 16
            for a deliberately small network, or 256 for large).

    Returns:
        tf.data.Dataset, tf.data.Dataset: train and test point with data
        augmentation.
    """

    # Our data can now be read into a `tf.data.Dataset()` object. We set the
    # shuffle buffer size to the entire size of the dataset as prior to this the
    # data is ordered by class. Data augmentation is important when working with
    # point cloud data. We create an augmentation function to jitter and shuffle
    # the train dataset.

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_points, test_labels))

    if len(train_points) > 0:
        train_dataset = train_dataset.shuffle(
            len(train_points)).map(data_transformation).batch(batch_size)
        train_dataset = \
            train_dataset.map(lambda points,
                              label: pointnet_preproc(points, label,
                                                      selected_points, knn_points),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size)
    test_dataset = \
        test_dataset.map(lambda points,
                         label: pointnet_preproc(points, label, selected_points,
                                                 knn_points),
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset
