# ******************************************************************************
# MIT License
#
# Copyright (c) 2017 Ngoc Anh Huynh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ******************************************************************************
"""
This module provides a method to generate YOLO anchors from dataset annotations.
"""

import random
import numpy as np


def _iou(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def _avg_iou(anns, centroids):
    n, _ = anns.shape
    s = 0.

    for i in range(anns.shape[0]):
        s += max(_iou(anns[i], centroids))

    return s / n


def _run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num) * (-1)
    iteration = 0

    indices = [random.randrange(ann_dims.shape[0]) for _ in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - _iou(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()


def generate_anchors(annotations_data, num_anchors=5, grid_size=(7, 7)):
    """ Creates anchors by clustering dimensions of the ground truth boxes
    from the training dataset.

    Args:
        annotations_data (dict): dictionnary of preprocessed VOC data
        num_anchors (int, optional): number of anchors
        grid_size (tuple, optional): size of the YOLO grid

    Returns:
        list: the computed anchors
    """
    annotation_dims = []

    # run k_mean to find the anchors
    for item in annotations_data:
        cell_w = item['image_shape'][0] / grid_size[0]
        cell_h = item['image_shape'][1] / grid_size[1]

        for box in item['boxes']:
            relative_w = float(box['x2'] - box['x1']) / cell_w
            relative_h = float(box['y2'] - box['y1']) / cell_h
            annotation_dims.append(tuple(map(float, (relative_w, relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = _run_kmeans(annotation_dims, num_anchors)
    print('\nAverage IOU for', num_anchors, 'anchors:',
          '%0.2f' % _avg_iou(annotation_dims, centroids))
    anchors = np.sort(np.round(centroids, 5), 0).tolist()
    print('Anchors: ', anchors)
    return anchors
