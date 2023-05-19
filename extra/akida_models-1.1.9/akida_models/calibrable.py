#!/usr/bin/env python
# ******************************************************************************
# Copyright 2022 Brainchip Holdings Ltd.
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
"""Tools for calibration"""

import tensorflow as tf
from quantizeml.layers import Calibrable


def get_objects(model, object_type):
    def _get_objects(layer, objects):
        if isinstance(layer, object_type):
            objects.append(layer)
        for attr in layer.__dict__.values():
            if isinstance(attr, tf.keras.layers.Layer):
                _get_objects(attr, objects)
    objects = []
    for layer in model.layers:
        _get_objects(layer, objects)
    return objects


def calibration_required(model):
    """Checks if a model must be calibrated at training time.

    Args:
        model (keras.Model): the model to check

    Returns:
        bool: True if a calibration is required before a train or tune action.
    """
    calibrables = get_objects(model, Calibrable)
    if len(calibrables) == 0:
        return False

    for calibrable in calibrables:
        # If the model has never been calibrated, all max_value of the Calibrable objects
        # will be set to 1.
        if tf.reduce_any(calibrable.variables[0] != 1):
            return False
    # all calibrable objects are unset
    return True
