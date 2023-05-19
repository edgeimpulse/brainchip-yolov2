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
This module defines a custom loss function for YOLO training.
"""

import numpy as np
import tensorflow as tf

from keras import backend as K

from .box_utils import compute_overlap

EPSILON = 1e-7


class YoloLoss():
    """ Computes YOLO loss from a model raw output.

        Custom loss calculation to be given to a Keras model "compile" method.
        YOLO loss uses sum-squared error and is composed of a localisation loss
        (using bounding box coordinates), a confidence loss (based on objects
        IoU) and a classification loss. Full details on the loss computation in
        the original paper https://arxiv.org/pdf/1506.02640.pdf.

        Args:
            anchors (list): anchors associated to the model.
            grid_size (tuple): yolo grid size.
            batch_size (int): the batch size.
            lambda_coord (float, optional): constant used to weight the loss
                of coordinate predictions.
            lambda_noobj (float, optional): constant used to weight the loss
                of empty boxes.
            lambda_obj (float, optional): constant used to weight the loss
                of non-empty boxes.
            lambda_class (float, optional): constant used to weight the
                classification loss.
            iou_filter (float, optional): IoU threshold used in the object loss.
    """

    def __init__(self,
                 anchors,
                 grid_size,
                 batch_size,
                 lambda_coord=1.0,
                 lambda_noobj=1.0,
                 lambda_obj=5.0,
                 lambda_class=1.0,
                 iou_filter=0.6):

        self.__name__ = 'yolo_loss'
        self.iou_filter = iou_filter

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

        self.batch_size = batch_size
        self.grid_size = grid_size
        self.nb_anchors = len(anchors)
        self.anchors = np.reshape(anchors, [1, 1, 1, self.nb_anchors, 2])

        self.c_grid = self._generate_yolo_grid(self.batch_size, self.grid_size,
                                               self.nb_anchors)

    @staticmethod
    def _generate_yolo_grid(batch_size, grid_size, nb_box):
        cell_x = tf.cast(
            tf.reshape(tf.tile(tf.range(grid_size[0]), [grid_size[1]]),
                       (1, grid_size[1], grid_size[0], 1, 1)), tf.float32)
        cell_y = tf.cast(
            tf.reshape(tf.tile(tf.range(grid_size[1]), [grid_size[0]]),
                       (1, grid_size[0], grid_size[1], 1, 1)), tf.float32)
        cell_y = tf.transpose(cell_y, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1),
                            [batch_size, 1, 1, nb_box, 1])
        return cell_grid

    def _transform_netout(self, y_pred_raw):

        y_pred_xy = K.sigmoid(y_pred_raw[..., :2]) + self.c_grid
        y_pred_wh = K.exp(y_pred_raw[..., 2:4]) * self.anchors
        y_pred_conf = K.sigmoid(y_pred_raw[..., 4:5])
        y_pred_class = y_pred_raw[..., 5:]

        return K.concatenate([y_pred_xy, y_pred_wh, y_pred_conf, y_pred_class],
                             axis=-1)

    def coord_loss(self, y_true, y_pred):
        """ Computes the localisation loss.

        Args:
            y_true (tf.Tensor): tensor of true labels
            y_pred (tf.Tensor): tensor of predicted labels

        Returns:
            float: the computed loss
        """

        b_xy_pred = y_pred[..., :2]
        b_wh_pred = y_pred[..., 2:4]

        b_xy = y_true[..., 0:2]
        b_wh = y_true[..., 2:4]

        indicator_coord = K.expand_dims(y_true[..., 4],
                                        axis=-1) * self.lambda_coord

        loss_xy = K.sum(K.square(b_xy - b_xy_pred) * indicator_coord)
        loss_wh = K.sum(K.square(b_wh - b_wh_pred) * indicator_coord)

        return (loss_wh + loss_xy) / 2

    def obj_loss(self, y_true, y_pred):
        """ Computes the confidence loss.

        Args:
            y_true (tf.Tensor): tensor of true labels
            y_pred (tf.Tensor): tensor of predicted labels

        Returns:
            float: the computed loss
        """

        b_o = y_true[..., 4]
        b_o_pred = y_pred[..., 4]

        num_true_labels = self.grid_size[0] * \
            self.grid_size[1] * self.nb_anchors
        y_true_p = K.reshape(y_true[..., :4],
                             shape=(self.batch_size, 1, 1, 1, num_true_labels,
                                    4))
        iou_scores_buff = compute_overlap(y_true_p, K.expand_dims(y_pred, axis=4))
        best_ious = K.max(iou_scores_buff, axis=4)

        indicator_noobj = K.cast(best_ious < self.iou_filter, np.float32) * (
            1 - y_true[..., 4]) * self.lambda_noobj
        indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_o = indicator_obj + indicator_noobj

        loss_obj = K.sum(K.square(b_o - b_o_pred) * indicator_o)
        return loss_obj / 2

    def class_loss(self, y_true, y_pred):
        """ Computes the classification loss.

        Args:
            y_true (tf.Tensor): tensor of true labels
            y_pred (tf.Tensor): tensor of predicted labels

        Returns:
            float: the computed loss
        """

        p_c_pred = K.softmax(y_pred[..., 5:])
        p_c = y_true[..., 5:]
        loss_class_arg = K.sum(K.square(p_c - p_c_pred), axis=-1)

        indicator_class = y_true[..., 4] * self.lambda_class

        loss_class = K.sum(loss_class_arg * indicator_class)

        return loss_class

    def __call__(self, y_true, y_pred_raw):

        y_pred = self._transform_netout(y_pred_raw)

        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)

        loss = total_coord_loss + total_obj_loss + total_class_loss

        return loss
