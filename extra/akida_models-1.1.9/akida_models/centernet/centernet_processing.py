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
Processing tools for CenterNet data handling.
"""
import numpy as np
from tensorflow.nn import max_pool2d

from ..detection.processing import BoundingBox


def decode_output(output,
                  nb_classes,
                  obj_threshold=0.1,
                  max_detections=100,
                  kernel=5):
    """ Decodes a CenterNet model.

    Args:
        output (tf.Tensor): model output to decode.
        nb_classes (int): number of classes.
        obj_threshold (float, optional): confidence threshold for a box. Defaults to 0.1.
        max_detection (int, optional): maximum number of boxes the model is allowed to produce.
            Defaults to 100.
        kernel (int, optional) : max pool kernel size. Defaults to 5.

    Returns:
        List: `BoundingBox` objects
    """

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    grid_h, grid_w = output.shape[:2]

    # Decode the output of the network
    center_heatmap_pred = _sigmoid(output[..., :nb_classes])
    wh_pred = output[..., nb_classes:nb_classes + 2]
    offset_pred = output[..., nb_classes + 2:nb_classes + 4]

    # Get local maximum
    hmax = max_pool2d(center_heatmap_pred[None, ...],
                      ksize=[kernel, kernel], strides=1, padding='SAME', data_format='NHWC')
    center_heatmap_pred[hmax[0] != center_heatmap_pred] = 0

    # Get top k from the heatmap
    perm_center_heatmap = np.transpose(center_heatmap_pred, (2, 0, 1))
    flattened_heatmap = np.reshape(perm_center_heatmap, (-1))

    topk_scores = np.partition(flattened_heatmap, -max_detections)[-max_detections:]
    topk_scores = np.flip(np.sort(topk_scores))
    topk_inds = np.argpartition(flattened_heatmap, -max_detections)[-max_detections:]
    topk_inds = topk_inds[np.argsort(flattened_heatmap[topk_inds])][::-1]
    topk_labels = topk_inds // (grid_h * grid_w)
    topk_inds = topk_inds % (grid_h * grid_w)
    topk_ys = topk_inds // grid_h
    topk_xs = topk_inds % grid_w

    # Transpose and gather features for the WH and OFFSET.
    # Removed the transpose as we don't do it above either
    wh_pred = np.reshape(wh_pred, [-1, wh_pred.shape[-1]])
    wh = wh_pred[topk_inds, ...]
    offset_pred = np.reshape(offset_pred, [-1, offset_pred.shape[-1]])
    offset = offset_pred[topk_inds, ...]

    # The output should be x,y,w,h
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = np.clip((topk_xs - wh[..., 0] / 2) / grid_w, a_min=0, a_max=grid_w)
    tl_y = np.clip((topk_ys - wh[..., 1] / 2) / grid_h, a_min=0, a_max=grid_h)
    br_x = np.clip((topk_xs + wh[..., 0] / 2) / grid_w, a_min=0, a_max=grid_w)
    br_y = np.clip((topk_ys + wh[..., 1] / 2) / grid_h, a_min=0, a_max=grid_h)

    boxes = []
    for i in range(max_detections):
        score = topk_scores[i]
        if score > obj_threshold:
            label = topk_labels[i]
            box = BoundingBox(tl_x[i], tl_y[i], br_x[i], br_y[i], score=score)
            box.label = label
            boxes.append(box)
        else:
            break

    return boxes
