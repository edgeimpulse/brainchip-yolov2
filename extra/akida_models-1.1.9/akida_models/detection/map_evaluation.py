#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Fizyr (https://fizyr.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Module used to compute mAP scores for YOLO classification.
"""

import keras
import numpy as np

from keras.utils import io_utils
from .processing import load_image, preprocess_image, decode_output
from .box_utils import compute_overlap


class MapEvaluation(keras.callbacks.Callback):
    """ Evaluate a given dataset using a given model.
        Code originally from https://github.com/fizyr/keras-retinanet.

        Args:
            model (keras.Model): model to evaluate.
            val_data (dict): dictionary containing validation data as obtained
             using `preprocess_widerface.py` module
            labels (list): list of labels as strings
            anchors (list): list of anchors boxes
            period (int, optional): periodicity the precision is printed,
             defaults to once per epoch.
            obj_threshold (float, optional): confidence threshold for a box
            nms_threshold (float, optional): non-maximal supression threshold
            max_box_per_image (int, optional): maximum number of detections per
             image
            is_keras_model (bool, optional): indicated if the model is a Keras
             model (True) or an Akida model (False)
            decode_output_fn (Callable, optional): function to decode model's outputs.
             Defaults to :func:`decode_output` (yolo decode output function).

        Returns:
            A dict mapping class names to mAP scores.
    """

    def __init__(self,
                 model,
                 val_data,
                 labels,
                 anchors,
                 period=1,
                 obj_threshold=0.5,
                 nms_threshold=0.5,
                 max_box_per_image=10,
                 is_keras_model=True,
                 decode_output_fn=decode_output):

        super().__init__()
        self._model = model
        self._data = val_data
        self._data_len = len(val_data)
        self._labels = labels
        self._num_classes = len(labels)
        self._anchors = anchors
        self._period = period
        self._obj_threshold = obj_threshold
        self._nms_threshold = nms_threshold
        self._max_box_per_image = max_box_per_image
        self._is_keras_model = is_keras_model
        self._decode_output = decode_output_fn

    def on_epoch_end(self, epoch, logs=None):
        """ Keras callback called at the end of an epoch.

        Args:
            epoch (int): index of epoch.
            logs (dict, optional): metric results for this training epoch, and
                for the validation epoch if validation is performed. Validation
                result keys are prefixed with val. For training epoch, the
                values of the Model’s metrics are returned.
                Example: {‘loss’: 0.2, ‘acc’: 0.7}. Defaults to None.
        """
        epoch += 1
        if self._period != 0 and (epoch % self._period == 0 or
                                  epoch == self.params.get('epochs', -1)):
            _map, average_precisions = self.evaluate_map()
            io_utils.print_msg("")
            for label, average_precision in average_precisions.items():
                io_utils.print_msg(self._labels[label] + ' {:.4f}'.format(average_precision))
            io_utils.print_msg('mAP: {:.4f}\n'.format(_map))
            logs.update({'map': _map})

    def evaluate_map(self):
        """ Evaluates current mAP score on the model.

        Returns:
            tuple: global mAP score and dictionnary of label and mAP for each
            class.
        """
        average_precisions = self._calc_avg_precisions()
        _map = sum(average_precisions.values()) / len(average_precisions)

        return _map, average_precisions

    def _load_annotations(self, i):
        annots = []

        for obj in self._data[i]['boxes']:
            annot = [
                obj['x1'], obj['y1'], obj['x2'], obj['y2'],
                self._labels.index(obj['label'])
            ]
            annots += [annot]

        return np.array(annots)

    def _calc_avg_precisions(self):

        # gather all detections and annotations
        all_detections = [[None
                           for _ in range(self._num_classes)]
                          for _ in range(self._data_len)]
        all_annotations = [[None
                            for _ in range(self._num_classes)]
                           for _ in range(self._data_len)]

        for i in range(self._data_len):
            raw_image = load_image(self._data[i]['image_path'])
            raw_height, raw_width, _ = raw_image.shape

            if self._is_keras_model:
                image = preprocess_image(raw_image, self._model.input_shape[1:])
                input_image = image[np.newaxis, :]
                output = self._model.predict(input_image)[0]
            else:
                image = preprocess_image(raw_image,
                                         self._model.layers[0].input_dims)
                input_image = image[np.newaxis, :].astype(np.uint8)
                potentials = self._model.predict(input_image)[0]

                h, w, _ = potentials.shape
                output = potentials.reshape(
                    (h, w, len(self._anchors), 4 + 1 + self._num_classes))

            pred_boxes = self._decode_output(output, self._anchors, self._num_classes,
                                             self._obj_threshold, self._nms_threshold)

            score = np.array([box.get_score() for box in pred_boxes])
            pred_labels = np.array([box.get_label() for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[
                    box.x1 * raw_width, box.y1 * raw_height, box.x2 * raw_width,
                    box.y2 * raw_height, box.score
                ] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # limit the number of predictions to max_box_per_image based on
            # score
            number_of_predictions = pred_boxes.shape[0]
            if number_of_predictions > self._max_box_per_image:
                pred_labels = pred_labels[:self._max_box_per_image]
                pred_boxes = pred_boxes[:self._max_box_per_image, :]

            # copy detections to all_detections
            for label in range(self._num_classes):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self._load_annotations(i)

            # copy ground truth to all_annotations
            for label in range(self._num_classes):
                all_annotations[i][label] = annotations[annotations[:, 4] ==
                                                        label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(self._num_classes):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._data_len):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations,
                                               mode="outer_product", box_format="xyxy")
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(
                true_positives + false_positives,
                np.finfo(np.float64).eps)

            # compute average precision
            average_precision = self._compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    @staticmethod
    def _compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision
        curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        Args:
            recall (list): the recall curve
            precision (list): the precision curve

        Returns:
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
