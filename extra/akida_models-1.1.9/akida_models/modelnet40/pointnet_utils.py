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
Utils for PointNet.
"""

import tensorflow as tf


def shift_to_7bit(values, val_min, val_max):
    """ Normalize data to 7 bits
    Args:
        values (tf.Tensor): input data to normalize.
        val_min (int): minimum input data value.
        val_max (int): maximum input data value.
    Returns:
        tf.Tensor: data normalized to 7 bits.
    """
    return tf.clip_by_value(
        tf.round((values - val_min) * (127 / (val_max - val_min))), 0, 127)


def get_reshape_factor(knn_points):
    """Obtains the reshape factor from number of points to include in each
    localised group.

    Args:
        knn_points (int): number of points to include in each localised group.
            Must be a power of 2, from 4 to 512.

    Returns:
        int: the computed reshape factor.
    """
    if knn_points in [4, 8]:
        reshape_factor = 2
    elif knn_points in [16, 32]:
        reshape_factor = 4
    elif knn_points in [64, 128]:
        reshape_factor = 8
    elif knn_points in [256, 512]:
        reshape_factor = 16
    else:
        raise ValueError("'knn_points' argument must be a power of 2, from 4 "
                         f"to 512 but received {knn_points}.")
    return reshape_factor


def pointnet_preproc(points, label, selected_points, knn_points):
    """ PointNet++ preprocessing function for sampling and grouping points.

    The preprocessing function is called for every point cloud to evaluate as
    it creates the input vector to our model from a point cloud.
    It has been empirically proven by the authors of the point cloud deep
    learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)
    that sampling a few points from the point cloud was enough to provide an
    input vector from which meaningful features could be extracted.

    In the point sampling and grouping operation:
    - a fixed number of points are selected from input points which defines the
      centroids of local regions.
    - the grouping process consists of identifying the set of points nearest to
      a centroid, and expressing their position relative to that centroid.

    Args:
        points (tf.Tensor): array representing a normalized point cloud.
        label (tf.Tensor): the labels.
        selected_points (int): number of points to process per sample.
        knn_points (int): number of neighboring points for the centroids to
            include in each localised group.

    Returns:
        tf.Tensor, tf.Tensor: the points preprocessed and labels.
    """

    input_points = points.shape[1]
    num_features = points.shape[-1]

    # The data loaded from file is normalized to the range [-1, 1]
    # However, for Akida, we need the inputs to be in the uint8 range (0, 255)
    # Also, the preprocessing function performs a substraction between point
    # coordinates, so some values are guaranteed to be negative at that point
    # in the pipeline.
    points = shift_to_7bit(points, -1.0, 1.0)

    # Random point selector
    # Simplification: selects same indices within a batch, and doesn't do
    # farthest point sampling
    sampler = tf.random.shuffle(tf.range(input_points),
                                seed=40)[:selected_points]
    inputs = tf.gather(points, sampler, axis=1, batch_dims=0)

    # Expects input of batch_size x sample_points x features
    # (with x, y, z, as the first 3 features)
    # batch_size x 1 x XYZ x sample_points
    temp1 = tf.expand_dims(tf.transpose(inputs[:, :, :3], perm=(0, 2, 1)), 1)
    # batch_size x selected_point x XYZ x 1
    temp2 = tf.expand_dims(inputs[:, :, :3], -1)
    points_dists = tf.reduce_sum(tf.square(temp1 - temp2), axis=2)
    # Note, LOWEST k, so top_k on negative values
    dists, points_idx = tf.math.top_k(-points_dists, k=knn_points, sorted=True)
    dists = -dists
    # dists and points_idx are batch_size x sample_points x knn
    # points_idx is size batch x selected_points x knn
    points_idx = tf.reshape(points_idx, [-1, selected_points * knn_points])
    # resized to batch x (selected_points * knn) x 1
    # Use that to gather from batch x all_points x features
    outpoints = tf.gather(inputs, points_idx, axis=1, batch_dims=1)
    outpoints = tf.reshape(outpoints,
                           [-1, selected_points, knn_points, num_features])
    # B x selected_points x 1 x 3
    points_center = tf.expand_dims(inputs[:, :selected_points, :3], axis=2)

    # Keep coordinates of original points
    subtraction_helper_l1 = tf.zeros_like(points_center)
    subtraction_helper_l2_to_end = tf.repeat(points_center,
                                             knn_points - 1,
                                             axis=2)
    subtraction_helper = tf.concat(
        [subtraction_helper_l1, subtraction_helper_l2_to_end], 2)

    if num_features > 3:
        sh = tf.shape(outpoints)
        subtraction_padding = tf.zeros((sh[0], sh[1], 1, sh[3] - 3))
        subtraction_helper = tf.concat(
            [subtraction_helper, subtraction_padding], -1)

    outpoints = (outpoints - subtraction_helper) / 127.0

    reshape_factor = get_reshape_factor(knn_points)

    # put the dimension to be folded last
    x = tf.transpose(outpoints, perm=[0, 3, 1, 2])
    # fold
    x = tf.reshape(x, [
        -1, num_features, selected_points * reshape_factor,
        int(knn_points / reshape_factor)
    ])
    # put dimensions back to original order
    points = tf.transpose(x, perm=[0, 2, 3, 1])
    return points, label
