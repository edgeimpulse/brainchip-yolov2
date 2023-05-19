#!/usr/bin/env python
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
Data generator for YOLO training.
"""

import copy
import cv2
import numpy as np

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox as BoundingBoxAug
from keras.utils.data_utils import Sequence

from .processing import BoundingBox


class BatchGenerator(Sequence):
    """ Data generator used for the training process.
    """

    def __init__(self,
                 input_shape,
                 data,
                 grid_size,
                 labels,
                 batch_size,
                 shuffle=True,
                 jitter=True):
        self._input_shape = input_shape
        self._data = data
        self._data_length = len(data)
        self._grid_size = grid_size
        self._labels = labels
        self._batch_size = batch_size

        self._shuffle = shuffle
        self._jitter = jitter

        self._aug_pipe = self.build_aug_pipeline()

        if shuffle:
            np.random.shuffle(self._data)

    @staticmethod
    def build_aug_pipeline():
        """ Define a sequence of augmentation steps that will be applied to every image.

        Returns:
            iaa.Sequential: sequence of augmentation.
        """
        # augmentors by https://github.com/aleju/imgaug
        def sometimes(aug): return iaa.Sometimes(0.5, aug)

        # All augmenters with per_channel=0.5 will sample one value per
        # image in 50% of all cases. In all other cases they will sample new
        # values per channel.
        return iaa.Sequential(
            [
                # apply the following augmenters to most images
                sometimes(iaa.Affine()),
                # execute 0 to 5 of the following (less important) augmenters
                # per image. Don't execute all of them, as that would often be
                # way too strong
                iaa.SomeOf(
                    (0, 5),
                    [
                        iaa.OneOf([
                            # blur images with a sigma between 0 and 3.0
                            iaa.GaussianBlur((0, 3.0)),
                            # blur image using local means (kernel sizes between
                            # 2 and 7)
                            iaa.AverageBlur(k=(2, 7)),
                            # blur image using local medians (kernel sizes
                            # between 3 and 11)
                            iaa.MedianBlur(k=(3, 11)),
                        ]),
                        # sharpen images
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        # add gaussian noise
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                        # randomly remove up to 10% of the pixels
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        ]),
                        # change brightness of images
                        iaa.Add((-10, 10), per_channel=0.5),
                        # change brightness of images
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        # improve or worsen the contrast
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    ],
                    random_order=True)
            ],
            random_order=True)

    def list2bbox(self, boxes, image_shape):
        """ Transforms bounding boxes from numpy array to BoundingBox format from imgaug library.
        """
        formattedbox = [BoundingBoxAug(x1=box["x1"], y1=box["y1"], x2=box["x2"], y2=box["y2"],
                        label=box['label']) for box in boxes]
        return BoundingBoxesOnImage(formattedbox, shape=image_shape)

    def bbox2list(self, boxes):
        """ Extracts bounding boxes from imgaug object to numpy array.
        """
        return [{"label": box.label, "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}
                for box in boxes]

    def __len__(self):
        return int(np.ceil(float(self._data_length) / self._batch_size))

    def __getitem__(self, idx):
        raise NotImplementedError("This function must be defined by the children class.")

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._data)

    def aug_image(self, train_instance, jitter):
        """ Performs data augmentation on an image.

        Args:
            train_instance (dict): the dictionnary of input data (image and
             boxes)
            jitter (bool): enable jitter in the augmentation pipeline

        Raises:
            ValueError: if the number of channel in the input shape is wrong
             (ie. different from 1 or 3).
            RuntimeError: if the image cannot be found

        Returns:
            tuple: the augmented image and the associated boxes
        """
        image_path = train_instance['image_path']
        if self._input_shape[2] == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif self._input_shape[2] == 3:
            image = cv2.imread(image_path)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None:
            raise RuntimeError(f'Cannot find {image_path}')

        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(train_instance['boxes'])

        if jitter:
            # scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            # translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy:(offy + h), offx:(offx + w)]

            # flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            # augment both image and boxes
            bbs = self.list2bbox(all_objs, image.shape)
            image, bbs = self._aug_pipe(image=image, bounding_boxes=bbs)
            bbs.remove_out_of_image().clip_out_of_image()
            all_objs = self.bbox2list(bbs)

        # resize the image to standard size
        image = cv2.resize(image, (self._input_shape[0], self._input_shape[1]))
        if self._input_shape[2] == 1:
            image = image[..., np.newaxis]
        image = image[..., ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['x1', 'x2']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self._input_shape[0]) / w)
                obj[attr] = max(min(obj[attr], self._input_shape[0]), 0)

            for attr in ['y1', 'y2']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self._input_shape[1]) / h)
                obj[attr] = max(min(obj[attr], self._input_shape[1]), 0)

            if jitter and flip > 0.5:
                xmin = obj['x1']
                obj['x1'] = self._input_shape[0] - obj['x2']
                obj['x2'] = self._input_shape[0] - xmin
        return image, all_objs


class BatchYoloGenerator(BatchGenerator):
    def __init__(self,
                 input_shape,
                 data,
                 grid_size,
                 labels,
                 anchors,
                 batch_size,
                 shuffle=True,
                 jitter=True):
        super().__init__(input_shape, data, grid_size, labels, batch_size, shuffle, jitter)
        self._anchors = [BoundingBox(0, 0, anchors[i][0], anchors[i][1])
                         for i in range(len(anchors))]

    def __getitem__(self, idx):
        lower_bound = idx * self._batch_size
        upper_bound = (idx + 1) * self._batch_size

        if upper_bound > self._data_length:
            upper_bound = self._data_length
            lower_bound = upper_bound - self._batch_size

        instance_count = 0

        x_batch = np.zeros((upper_bound - lower_bound, self._input_shape[1],
                            self._input_shape[0], self._input_shape[2]))

        y_batch = np.zeros(
            (upper_bound - lower_bound, self._grid_size[1], self._grid_size[0],
             len(self._anchors),
             4 + 1 + len(self._labels)))  # desired network output

        for train_instance in self._data[lower_bound:upper_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self._jitter)

            for obj in all_objs:
                if obj['x2'] > obj['x1'] and obj['y2'] > obj['y1'] and obj[
                        'label'] in self._labels:
                    center_x = .5 * (obj['x1'] + obj['x2'])
                    center_x = center_x / (float(self._input_shape[0]) /
                                           self._grid_size[0])
                    center_y = .5 * (obj['y1'] + obj['y2'])
                    center_y = center_y / (float(self._input_shape[1]) /
                                           self._grid_size[0])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self._grid_size[0] and grid_y < self._grid_size[
                            1]:
                        obj_indx = self._labels.index(obj['label'])

                        center_w = (obj['x2'] - obj['x1']) / (
                            float(self._input_shape[0]) / self._grid_size[0])
                        center_h = (obj['y2'] - obj['y1']) / (
                            float(self._input_shape[1]) / self._grid_size[0])

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundingBox(0, 0, center_w, center_h)

                        for anchor_id, anchor in enumerate(self._anchors):
                            iou = shifted_box.iou(anchor)

                            if max_iou < iou:
                                best_anchor = anchor_id
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class
                        # probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor,
                                0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor,
                                4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor,
                                5 + obj_indx] = 1

            # assign input image to x_batch
            x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch
