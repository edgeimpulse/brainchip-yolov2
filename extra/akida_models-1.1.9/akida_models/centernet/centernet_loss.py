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
This module defines a custom loss function for CenterNet training
"""
from keras import backend as K
import tensorflow as tf


class CenternetLoss():
    """ Computes CenterNet loss from a model raw output.

    Args:
        alpha (float, optional): alpha parameter in heatmap loss. Defaults to 2.0.
        gamma (float, optional): gamma parameter in heatmap loss. Defaults to 4.0.
        eps (float, optional): epsilon parameter in heatmap loss. Defaults to 1e-12.
        heatmap_loss_weight (float, optional): heatmap loss weight. Defaults to 1.0.
        wh_loss_weight (float, optional): location loss weight. Defaults to 0.1.
        offset_loss_weight (float, optional): offset loss weight. Defaults to 1.0.

    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 eps=1e-12,
                 heatmap_loss_weight=1.0,
                 wh_loss_weight=0.1,
                 offset_loss_weight=1.0):
        # Parameters for the gaussian focal loss for the heatmap branch
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps

        # Loss weight parameters
        self.heatmap_loss_weight = heatmap_loss_weight
        self.wh_loss_weight = wh_loss_weight
        self.offset_loss_weight = offset_loss_weight

    def _transform_netout(self, y_pred_raw):
        """Transforms the output of the network:

        - cast to float32
        - extracts the // wh, offset and heatmap from fused map if necessary
        - applies sigmoid to the heatmap prediction

        Args:
            y_pred_raw (tf.Tensor): raw network predictions.

        Returns:
            tuple of tf.Tensor: Predictions transformed on xy, wh and offset values.
        """
        y_pred_raw = tf.cast(y_pred_raw, dtype=tf.float32)
        y_pred_xy = K.sigmoid(y_pred_raw[..., :-4])
        y_pred_wh = y_pred_raw[..., -4:-2]
        y_pred_offset = y_pred_raw[..., -2:]
        return y_pred_xy, y_pred_wh, y_pred_offset

    def _get_targets(self, y_true):
        """Extract ground truth for each branch and compute avg_factor, wh_offset_target_weight
        here so we don't have to pass it through the whole model

        Args:
            y_true (tf.Tensor): ground truth.

        Returns:
            tuple of tf.Tensor: labels in xy, wh, offset, avg_factor and wh_offset format.
        """
        target_xy = y_true[..., :-4]
        target_wh = y_true[..., -4:-2]
        target_offset = y_true[..., -2:]

        # Extract the average factor counts the number of targets to be learned
        # max(1, center_heatmap_target.eq(1).sum())
        tmp = tf.equal(tf.constant(1.0, dtype=y_true.dtype), target_xy)
        tmp = tf.cast(tmp, dtype=tf.float32)
        tmp = tf.reduce_sum(tmp)
        avg_factor = tf.reduce_max([tf.constant(1.0, dtype=tmp.dtype), tmp])

        # Extract the wh offset target weight
        # wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1
        # => 1 anywhere there is a target offset and wh
        tmp = tf.equal(tf.constant(0, dtype=y_true.dtype), target_offset)
        tmp = tf.logical_not(tmp)
        wh_offset_target_weight = tf.cast(tmp, dtype=tf.float32)
        return target_xy, target_wh, target_offset, avg_factor, wh_offset_target_weight

    def heatmap_loss(self, y_true, y_pred, avg_factor):
        """Implements `Gaussian Focal loss <https://arxiv.org/abs/1708.02002>`_
        for targets in gaussian distribution.

        Original source: mmdetection/losses/gaussian_focal_loss

        Args:
            y_true (tf.Tensor): tensor of true labels.
            y_pred (tf.Tensor): tensor of predicted labels.
            avg_factor (tf.Tensor): average factor.

        Returns:
            tf.Tensor: Heatmap loss
        """
        # Compute the loss
        pos_weights = tf.cast(tf.equal(y_true, 1.0), dtype=tf.float32)
        neg_weights = tf.math.pow((1 - y_true), self._gamma)
        pos_loss = -tf.math.log(y_pred + self._eps) * \
            tf.math.pow((1 - y_pred), self._alpha) * pos_weights
        neg_loss = -tf.math.log(1 - y_pred + self._eps) * \
            tf.math.pow(y_pred, self._alpha) * neg_weights
        loss = pos_loss + neg_loss
        # Compute the average across the matrix
        loss = tf.reduce_sum(loss) / avg_factor
        return loss

    def l1_loss(self, y_true, y_pred, avg_factor, weights=None):
        """L1 loss, used in location loss

        Args:
            y_true (tf.Tensor): tensor of true labels.
            y_pred (tf.Tensor): tensor of predicted labels.
            avg_factor (tf.Tensor): average factor.
            weights (tf.Tensor, optional): factor to multiply the loss. Defaults to None.

        Returns:
            tf.Tensor: L1 loss
        """
        difference = y_true - y_pred
        loss = tf.abs(difference)
        if weights is not None:
            loss *= weights
        loss = tf.reduce_sum(loss) / avg_factor
        return loss

    def __call__(self, y_true, y_pred_raw):
        # Get the avg factor and wh / offset weights
        (target_xy, target_wh, target_offset, avg_factor,
            wh_offset_target_weight) = self._get_targets(y_true)
        # Extract the 3 // branches + apply sigmoid
        y_pred_xy, y_pred_wh, y_pred_offset = self._transform_netout(y_pred_raw)

        # Heatmap loss
        center_heatmap_loss = self.heatmap_loss(target_xy, y_pred_xy, avg_factor)
        center_heatmap_loss *= self.heatmap_loss_weight

        # Wh loss
        wh_loss = self.l1_loss(target_wh, y_pred_wh, avg_factor * 2, wh_offset_target_weight)
        wh_loss *= self.wh_loss_weight

        # Offset loss
        offset_loss = self.l1_loss(target_offset, y_pred_offset, avg_factor * 2,
                                   wh_offset_target_weight)
        offset_loss *= self.offset_loss_weight

        loss = center_heatmap_loss + wh_loss + offset_loss
        return loss
