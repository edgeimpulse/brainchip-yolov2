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
"""
ImageNet1K dataset mean and std constants definition.
"""

import warnings


# Taken from timm.data.constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def obtain_input_shape(input_shape, default_size, min_size, include_top):
    """Internal utility to compute/validate a model's input shape.

    Args:
        input_shape: either None (will return the default model input shape),
            or a user-provided shape to be validated.
        default_size: default input width/height for the model.
        min_size: minimum input width/height accepted by the model.
        include_top: whether the model is expected to
            be linked to a classifier via a Flatten layer.

    Returns:
        tuple of integers: input shape (may include None entries).

    Raises:
        ValueError: in case of invalid argument values.
    """
    if input_shape and len(input_shape) == 3:
        if input_shape[-1] not in {1, 3}:
            warnings.warn('This model usually expects 1 or 3 input channels. '
                          'However, it was passed an input_shape with ' +
                          str(input_shape[-1]) + ' input channels.')
        default_shape = (default_size, default_size, input_shape[-1])
    else:
        default_shape = (default_size, default_size, 3)

    if input_shape:
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError(
                    '`input_shape` must be a tuple of three integers.')
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) +
                                 '; got `input_shape=' + str(input_shape) + '`')
    else:
        if include_top:
            input_shape = default_shape
        else:
            input_shape = (None, None, 3)
    if include_top:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape
