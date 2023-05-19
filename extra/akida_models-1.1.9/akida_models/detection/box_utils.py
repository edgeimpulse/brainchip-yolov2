#!/usr/bin/env python
# ******************************************************************************
# Copyright 2023 Brainchip Holdings Ltd.
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
This module provides some utils to handle boxes and calculate the jaccard overlap score.
"""
import numpy as np
import tensorflow as tf


def xywh_to_xyxy(boxes):
    """Convert a set of boxes from format xywh to xyxy, where each format represent:

        * 'xywh': format of ('cx', 'xy', 'w', 'h'), also called 'centroids' and
        * 'xyxy': format of ('x_min', 'y_min', 'x_max', 'y_max'), also called 'corners'.

    Args:
        boxes (tf.tensor or np.ndarray): tensor with shape (N, 4)

    Returns:
        tf.tensor or np.ndarray: tensor with new format
    """
    assert boxes.shape[-1] == 4, "Expected 4 as last dimension."
    x1y1 = boxes[..., :2] - 0.5 * boxes[..., -2:]
    x2y2 = boxes[..., :2] + 0.5 * boxes[..., -2:]
    if isinstance(boxes, tf.Tensor):
        return tf.concat([x1y1, x2y2], axis=-1)
    return np.concatenate([x1y1, x2y2], axis=-1)


def xyxy_to_xywh(boxes):
    """Convert a set of boxes from format xyxy to xywh, where each format represent:

        * 'xyxy': format of ('x_min', 'y_min', 'x_max', 'y_max'), also called 'corners' and
        * 'xywh': format of ('cx', 'xy', 'w', 'h'), also called 'centroids'.

    Args:
        boxes (tf.tensor): tensor with shape (N, 4)

    Returns:
        tf.tensor: tensor with new format
    """
    assert boxes.shape[-1] == 4, "Expected 4 as last dimension."
    tf.debugging.assert_greater_equal(boxes[..., -2:], boxes[..., :2],
                                      "x2y2 must be greater than x1y1")
    wh = boxes[..., -2:] - boxes[..., :2]
    xy = boxes[..., :2] + 0.5 * wh
    if isinstance(boxes, tf.Tensor):
        return tf.concat([xy, wh], axis=-1)
    return np.concatenate([xy, wh], axis=-1)


def compute_overlap(a1, a2, mode="element_wise", box_format="xywh"):
    """Calculate ious between a1, a2 in two different modes:

        * element_wise: compute iou element-by-element, returning 1D array tensor,
        * outer_product: compute cross iou with all possible combination between inputs.

    Args:
        a1 (tf.tensor or np.ndarray): set of boxes, with shape at least equal to (N, 4).
        a2 (tf.tensor or np.ndarray): set of boxes, with compatible broadcast-shape
            (in 'element_wise' mode) or shape at least equal to (N, 4) (in 'outer_product' mode).
        box_format (str, optional): format of both inputs. Defaults to 'xywh'.

    Returns:
        tf.tensor or np.ndarray: IoU between inputs with shape (N,) in 'element_wise',
            otherwise (N, M).
    """
    format_choices = ["xywh", "xyxy"]
    mode_choises = ["element_wise", "outer_product"]
    assert box_format in format_choices, f"box format must be one of {format_choices}"
    assert mode in mode_choises, f"mode must be one of {mode_choises}"

    def process_boxes(a):
        # Decompose the input set of boxes into three parts:
        # [(x_min, y_min), (x_max, y_max), (w, h)]
        if box_format == "xywh":
            a_wh = a[..., 2:4]
            a = xywh_to_xyxy(a[..., :4])
        else:
            a_wh = a[..., 2:4] - a[..., :2]
        return a[..., :2], a[..., 2:4], a_wh

    # Check inputs are not empty
    a1_num_boxes = a1.shape[0]
    a2_num_boxes = a2.shape[0]
    is_tf_tensor = isinstance(a1, tf.Tensor)
    if a1_num_boxes == 0 or a2_num_boxes == 0:
        if is_tf_tensor:
            return tf.zeros((a1_num_boxes, a2_num_boxes))
        return np.zeros((a1_num_boxes, a2_num_boxes))

    # In outer product, we need expand tensors. Broadcast will do the rest of the job.
    if mode == "outer_product":
        a1 = a1[:, None]
        a2 = a2[None]

    # Process two sets
    a2_mins, a2_maxes, a2_wh = process_boxes(a2)
    a1_mins, a1_maxes, a1_wh = process_boxes(a1)

    # Intersection as min((a2_maxes, a1_maxes) - max(a2_mins, a1_mins)
    intersect_mins = tf.math.maximum(a2_mins, a1_mins)
    intersect_maxes = tf.math.minimum(a2_maxes, a1_maxes)

    # Getting the intersections in the xy (aka the width, height intersection)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0)

    # Multiply to get intersecting area
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Values for the single sets
    true_areas = a1_wh[..., 0] * a1_wh[..., 1]
    pred_areas = a2_wh[..., 0] * a2_wh[..., 1]

    # Compute union for the IoU
    add_areas = tf.convert_to_tensor(pred_areas + true_areas, dtype=intersect_areas.dtype)
    union_areas = add_areas - intersect_areas
    iou = intersect_areas / union_areas

    if is_tf_tensor:
        return iou
    return iou.numpy()
