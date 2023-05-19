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
Data generator for CenterNet training
"""

import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

from ..detection.batch_generator import BatchGenerator


class BatchCenternetGenerator(BatchGenerator):
    """
    Data generator used for the training process
    """
    @staticmethod
    def build_aug_pipeline():
        # augmentors by https://github.com/aleju/imgaug
        def sometimes(aug): return iaa.Sometimes(0.5, aug)

        # All augmenters with per_channel=0.5 will sample one value per
        # image in 50% of all cases. In all other cases they will sample new
        # values per channel.
        return iaa.Sequential(
            [
                # apply the following augmenters to most images
                sometimes(iaa.Affine()),
                sometimes(iaa.Cutout(nb_iterations=(0, 3), position="uniform", size=(0, 0.3))),
                # execute 0 to 5 of the following (less important) augmenters
                # per image. Don't execute all of them, as that would often be
                # way too strong
                iaa.SomeOf(
                    (0, 5),
                    [
                        iaa.OneOf([
                            # blur images with a sigma between 0 and 3.0
                            iaa.GaussianBlur((0, 3.0)),
                            # blur image using local means (kernel sizes between 2 and 7)
                            iaa.AverageBlur(k=(2, 7)),
                            # blur image using local medians (kernel sizes between 3 and 11)
                            iaa.MedianBlur(k=(3, 11)),
                        ]),
                        # sharpen images
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                        # randomly remove up to 10% of the pixels
                        iaa.OneOf([iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout((0.01, 0.05), size_percent=0.5)]),
                        # change brightness of images
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.OneOf([
                            iaa.pillike.Equalize(),
                            iaa.pillike.Autocontrast()
                        ])
                    ], random_order=True)
            ],
            random_order=True)

    def __getitem__(self, idx):
        lower_bound = idx * self._batch_size
        upper_bound = (idx + 1) * self._batch_size

        if upper_bound > self._data_length:
            upper_bound = self._data_length
            lower_bound = upper_bound - self._batch_size

        instance_count = 0

        N = upper_bound - lower_bound
        x_batch = np.zeros((N, self._input_shape[1], self._input_shape[0], self._input_shape[2]))
        y_batch = np.zeros((N, self._grid_size[1], self._grid_size[0], 2 + 2 + len(self._labels)))

        for train_instance in self._data[lower_bound:upper_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self._jitter)

            for obj in all_objs:
                if obj['x2'] > obj['x1'] and obj['y2'] > obj['y1'] and obj['label'] in self._labels:
                    # find center
                    center_x = .5 * (obj['x1'] + obj['x2'])
                    # express it in output size
                    center_x = center_x / (float(self._input_shape[0]) / self._grid_size[0])
                    center_y = .5 * (obj['y1'] + obj['y2'])
                    center_y = center_y / (float(self._input_shape[1]) / self._grid_size[0])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self._grid_size[0] and grid_y < self._grid_size[1]:
                        obj_indx = self._labels.index(obj['label'])
                        scale_box_w = (obj['x2'] - obj['x1']) / \
                            (float(self._input_shape[0]) / self._grid_size[0])
                        scale_box_h = (obj['y2'] - obj['y1']) / (
                            float(self._input_shape[1]) / self._grid_size[0])

                        # get the center point and use a gaussian kernel as the target
                        radius = self._gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)

                        # check that the radius is positive
                        radius = np.max(int(radius), 0)
                        heatmap = np.zeros(
                            (self._grid_size[0], self._grid_size[1], len(self._labels)))
                        heatmap = self._gen_gaussian_target(
                            heatmap, [grid_y, grid_x], obj_indx, radius)

                        # assign ground truth to y_batch
                        y_batch[instance_count, ..., :len(self._labels)] += heatmap
                        y_batch[instance_count, grid_y, grid_x, -4] = scale_box_w
                        y_batch[instance_count, grid_y, grid_x, -3] = scale_box_h
                        y_batch[instance_count, grid_y, grid_x, -2] = center_x - grid_x
                        y_batch[instance_count, grid_y, grid_x, -1] = center_y - grid_y

            # assign input image to x_batch
            x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch

    @staticmethod
    def _gaussian2D(radius, sigma=1, eps=tf.keras.backend.epsilon()):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int, optional): Sigma of gaussian function. Defaults to 1.
            eps (float, optional): Epsilon value. Defaults to 1e-7.

        Returns:
            np.array: Gaussian kernel with a ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        x = np.reshape(np.arange(-radius, radius + 1), [1, -1])
        y = np.reshape(np.arange(-radius, radius + 1), [-1, 1])
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        # Clamp smaller values to zero
        h[h < (eps * h.max())] = 0
        return h

    @staticmethod
    def _gen_gaussian_target(heatmap, center, obj_idx, radius):
        """Generate 2D gaussian heatmap.

        Args:
            heatmap (np.array): Input heatmap, the gaussian kernel will cover on it and maintain
                the max value.
            center (list[int]): Coordinates of gaussian kernel's center.
            obj_idx (int): The class index for the center point.
            radius (int): Radius of gaussian kernel.

        Returns:
            out_heatmap (np.array): Updated heatmap covered by gaussian kernel.

        Note:
            Taken from pytorch
        """
        diameter = 2 * radius + 1
        gaussian_kernel = BatchCenternetGenerator._gaussian2D(radius, sigma=diameter / 6)
        x, y = center
        height, width = heatmap.shape[:2]

        # Find the smallest value so that if the point is near the edge we don't end outside
        # (e.g. x = 3 and radius is 10, then we go from the x-3 to x+10)
        left = np.min([x, radius])
        right = np.min([width - x, radius + 1])
        top = np.min([y, radius])
        bottom = np.min([height - y, radius + 1])

        # Compare the gaussian kernel to the heatmap (in case there's already a point of
        # interest there) and keep the max value
        flattened_kernel = np.reshape(gaussian_kernel, [-1])

        # Range the dimensions
        i = 0
        for d0 in np.arange(y - top, y + bottom):
            for d1 in np.arange(x - left, x + right):
                heatmap[d1, d0, obj_idx] = flattened_kernel[i]
                i += 1
        return heatmap

    @staticmethod
    def _gaussian_radius(det_size, min_overlap):
        r"""Generate 2D gaussian radius.

        This function is modified from the `official github repo
        <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
        utils.py#L65>`_.

        Given ``min_overlap``, radius could computed by a quadratic equation
        according to Vieta's formulas.

        There are 3 cases for computing gaussian radius, details are following:

        - Case 1: one corner is inside the gt box and the other is outside.

        .. code:: text

            |<   width   >|

            lt-+----------+         -
            |  |          |         ^
            +--x----------+--+
            |  |          |  |
            |  |          |  |    height
            |  | overlap  |  |
            |  |          |  |
            |  |          |  |      v
            +--+---------br--+      -
            |          |  |
            +----------+--x

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
            {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
            {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h}
            {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

        - Case 2: both two corners are inside the gt box.

        .. code:: text

            |<   width   >|

            lt-+----------+         -
            |  |          |         ^
            +--x-------+  |
            |  |       |  |
            |  |overlap|  |       height
            |  |       |  |
            |  +-------x--+
            |          |  |         v
            +----------+-br         -

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
            {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
            {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h}
            {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

        - Case 3: both two corners are outside the gt box.

        .. code:: text

            |<   width   >|

            x--+----------------+
            |  |                |
            +-lt-------------+  |   -
            |  |             |  |   ^
            |  |             |  |
            |  |   overlap   |  | height
            |  |             |  |
            |  |             |  |   v
            |  +------------br--+   -
            |                |  |
            +----------------+--x

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
            {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
            {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
            {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

        Args:
            det_size (list[int]): Shape of object.
            min_overlap (float): Min IoU with ground truth for boxes generated by
                keypoints inside the gaussian kernel.

        Returns:
            radius (int): Radius of gaussian kernel.

        Notes:
            Explanation of figure: ``lt`` and ``br`` indicates the left-top and bottom-right
            corner of ground truth box. ``x`` indicates the generated corner at the limited
            position when ``radius=r``.
        """
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        return np.min([r1, r2, r3])
