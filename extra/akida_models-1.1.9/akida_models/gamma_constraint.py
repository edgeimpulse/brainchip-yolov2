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
Custom constraint for BatchNormalization layers and a method helper to apply the
constraint.
"""

import warnings

from keras.models import clone_model
from keras.utils.generic_utils import custom_object_scope
from keras.layers import BatchNormalization

from cnn2snn.min_value_constraint import MinValueConstraint


def add_gamma_constraint(model):
    """ Method helper to add a MinValueConstraint to an existing model so that
    gamma values of its BatchNormalization layers are above a defined minimum.

    This is typically used to help having a model that will be Akida compatible
    after conversion. In some cases, the mapping on hardware will fail because
    of huge values for `threshold` or `act_step` with a message indicating that
    a value cannot fit in a 20 bit signed or unsigned integer.
    In such a case, this helper can be called to apply a constraint that can fix
    the issue.

    Note that in order for the constraint to be applied to the actual weights,
    some training must be done: for an already trained model, it can be on a few
    batches, one epoch or more depending on the impact the constraint has on
    accuracy. This helper can also be called to a new model that has not been
    trained yet.

    Args:
        model (keras.Model): the model for which gamma constraints will be
            added.

    Returns:
        keras.Model: the same model with BatchNormalisation layers updated.
    """

    def apply_gamma_constraint(layer):
        constraint = MinValueConstraint()
        if isinstance(layer, BatchNormalization):
            if layer.gamma_constraint is not None:
                warnings.warn(
                    f"Layer {layer.name} already has a gamma_constraint set "
                    f"to {layer.gamma_constraint}, it will be overwritten.")
            bn = BatchNormalization.from_config(layer.get_config())
            bn.gamma_constraint = constraint
            return bn
        return layer.__class__.from_config(layer.get_config())

    with custom_object_scope({'MinValueConstraint': MinValueConstraint}):
        updated_model = clone_model(model,
                                    clone_function=apply_gamma_constraint)
    updated_model.set_weights(model.get_weights())

    return updated_model
