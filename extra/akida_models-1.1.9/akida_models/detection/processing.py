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
Processing tools for YOLO data handling.
"""

import os

import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf


class BoundingBox:
    """ Utility class to represent a bounding box.

    The box is defined by its top left corner (x1, y1), bottom right corner
    (x2, y2), label, score and classes.
    """

    def __init__(self, x1, y1, x2, y2, score=-1, classes=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = -1
        self.score = score
        self.classes = classes

    def __repr__(self):
        return "<BoundingBox({}, {}, {}, {}, {}, {}, {})>\n".format(
            self.x1, self.x2, self.y1, self.y2, self.get_label(),
            self.get_score(), self.classes)

    def get_label(self):
        """ Returns the label for this bounding box.

        Returns:
            Index of the label as an integer.
        """
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """ Returns the score for this bounding box.

        Returns:
            Confidence as a float.
        """
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

    def iou(self, other):
        """ Computes intersection over union ratio between this bounding box and
        another one.

        Args:
            other (BoundingBox): the other bounding box for IOU computation

        Returns:
            IOU value as a float
        """

        def _interval_overlap(interval_1, interval_2):
            x1, x2 = interval_1
            x3, x4 = interval_2

            if x3 < x1:
                if x4 < x1:
                    return 0
                return min(x2, x4) - x1
            if x2 < x3:
                return 0
            return min(x2, x4) - x3

        intersect_w = _interval_overlap([self.x1, self.x2],
                                        [other.x1, other.x2])
        intersect_h = _interval_overlap([self.y1, self.y2],
                                        [other.y1, other.y2])

        intersect = intersect_w * intersect_h

        w1, h1 = self.x2 - self.x1, self.y2 - self.y1
        w2, h2 = other.x2 - other.x1, other.y2 - other.y1

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union


def load_image(image_path):
    """ Loads an image from a path.

    Args:
        image_path (string): full path of the image to load

    Returns:
        a Tensorflow image Tensor
    """
    raw_image = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(raw_image, channels=3)


def preprocess_image(image_buffer, output_size):
    """ Preprocess an image for YOLO inference.

    Args:
        image_buffer (tf.Tensor): image to preprocess
        output_size (tuple): shape of the image after preprocessing

    Returns:
        A resized and normalized image as a Numpy array.
    """
    # Resize
    width = tf.constant(output_size[0])
    height = tf.constant(output_size[1])
    image = tf.compat.v1.image.resize(image_buffer, [height, width],
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)
    return image.numpy()


def decode_output(output,
                  anchors,
                  nb_classes,
                  obj_threshold=0.5,
                  nms_threshold=0.5):
    """ Decodes a YOLO model output.

    Args:
        output (tf.Tensor): model output to decode
        anchors (list): list of anchors boxes
        nb_classes (int): number of classes
        obj_threshold (float): confidence threshold for a box
        nms_threshold (float): non-maximal supression threshold

    Returns:
        List of `BoundingBox` objects
    """

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def _softmax(x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)

    grid_h, grid_w, nb_box = output.shape[:3]

    boxes = []

    # decode the output by the network
    output[..., 4] = _sigmoid(output[..., 4])
    output[...,
           5:] = output[..., 4][..., np.newaxis] * _softmax(output[..., 5:])
    output[..., 5:] *= output[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 5th element onwards are class classes
                classes = output[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = output[row, col, b, :4]

                    x = (col + _sigmoid(x)
                         ) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)
                         ) / grid_h  # center position, unit: image height
                    w = anchors[b][0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[b][1] * np.exp(h) / grid_h  # unit: image height

                    confidence = output[row, col, b, 4]

                    x1 = max(x - w / 2, 0)
                    y1 = max(y - h / 2, 0)
                    x2 = min(x + w / 2, grid_w)
                    y2 = min(y + h / 2, grid_h)

                    box = BoundingBox(x1, y1, x2, y2, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_classes):
        sorted_indices = list(
            reversed(np.argsort([box.classes[c] for box in boxes])))
        for ind, index_i in enumerate(sorted_indices):
            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(ind + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                # filter out redundant boxes (same class and overlapping too
                # much)
                if (boxes[index_i].iou(boxes[index_j]) >= nms_threshold) and (
                        c == boxes[index_i].get_label()) and (
                            c == boxes[index_j].get_label()):
                    boxes[index_j].score = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def parse_voc_annotations(gt_folder, image_folder, file_path, labels):
    """ Loads PASCAL-VOC data.

    Data is loaded using the groundtruth informations and stored in a
    dictionary.

    Args:
        gt_folder (str): path to the folder containing ground truth files
        image_folder (str): path to the folder containing the images
        file_path (str): file containing the list of files to parse
        labels (list): list of labels of interest
    Returns:
        dict: a dictionnary containing all data present in the ground truth file

    """

    def _build_list(file_path):
        """
        Builds a list of file names from a text file path.

        Args:
            file_path (string): path of the file containing the names
        Returns:
            list: list of file names
        """
        file_list = []
        with open(file_path, "r", encoding='UTF-8') as f:
            for line in f:
                file_list.append(line.rstrip())
        return file_list

    file_list = _build_list(file_path)

    all_data = []

    for f in file_list:
        tree = et.parse(os.path.join(gt_folder, f) + ".xml")

        data = {"boxes": []}

        for elem in tree.iter():
            if 'filename' in elem.tag:
                data['image_path'] = os.path.join(image_folder, elem.text)
            if 'size' in elem.tag:
                for attr in list(elem):
                    if 'width' in attr.tag:
                        width = int(attr.text)
                    if 'height' in attr.tag:
                        height = int(attr.text)
                    if 'depth' in attr.tag:
                        depth = int(attr.text)
                data['image_shape'] = (width, height, depth)

            if 'object' in elem.tag:
                box = {}

                for attr in list(elem):
                    if 'difficult' in attr.tag:
                        if int(attr.text) != 0:
                            box.clear()
                            break

                    if 'name' in attr.tag:
                        box['label'] = attr.text

                        if box['label'] not in labels:
                            box.clear()
                            break

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                box['x1'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                box['y1'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                box['x2'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                box['y2'] = int(round(float(dim.text)))

                if len(box) == 5:
                    data["boxes"].append(box)

        if len(data["boxes"]) != 0:
            all_data.append(data)

    return all_data


def parse_widerface_annotations(gt_file, image_folder):
    """ Loads WiderFace data.

    Data is loaded using the groundtruth informations and stored in a
    dictionary.

    Args:
        gt_file (str): path to the ground truth file
        image_folder (str): path to the directory containing the images
    Returns:
        dict: a dictionnary containing all data present in the ground truth file

    """

    def _is_valid_box(w_box, h_box, w_img, h_img):
        box_area = w_box * h_box
        img_area = w_img * h_img
        return box_area >= img_area / 60.0

    all_data = []

    with open(gt_file, "r", encoding='UTF-8') as f:
        for line in f:
            file_path = line.rstrip()
            full_file_path = os.path.join(image_folder, str(file_path))

            if not os.path.isfile(full_file_path):
                raise RuntimeError(f"Not a file path: {full_file_path}")

            image = load_image(full_file_path)
            shape = (image.shape[1], image.shape[0], image.shape[2])

            data = {
                "image_path": full_file_path,
                "image_shape": shape,
                "boxes": []
            }

            nb_boxes = int(f.readline())
            if nb_boxes == 0:
                f.readline()

            for _ in range(nb_boxes):
                annotation = f.readline()
                values = annotation.split(' ')

                x1 = int(values[0])
                y1 = int(values[1])
                w = int(values[2])
                h = int(values[3])

                if _is_valid_box(w, h, shape[0], shape[1]):
                    box = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x1 + w,
                        "y2": y1 + h,
                        "label": "face"
                    }
                    data["boxes"].append(box)

            if len(data["boxes"]) != 0:
                all_data.append(data)

        return all_data
