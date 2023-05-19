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
CenterNet model definition for detection
"""
import numpy as np

from keras import Model, initializers
from keras.layers import Conv2D


from .. import akidanet_imagenet
from ..layer_blocks import (separable_conv_block, conv_block, conv_transpose_block,
                            sepconv_transpose_block)


def centernet_base(input_shape=(224, 224, 3),
                   classes=2,
                   alpha=0.5,
                   input_scaling=(127, -1)):
    """ A Keras Model implementing the CenterNet architecture, on top of an AkidaNet backbone

    Args:
        input_shape (tuple, optional): input shape. Defaults to (224, 224, 3).
        classes (int, optional): number of output classes. Defaults to 2.
        alpha (float): controls the width of the model. Defaults to 0.5.
        input_scaling (tuple, optional): input scaling. Defaults to (127, -1).

    Returns:
        keras.Model: a Keras Model instance.
    """
    # Create an AkidaNet network without top layers
    base_model = akidanet_imagenet(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        input_scaling=input_scaling)

    x = base_model.layers[-1].output

    # Build the neck with up convolutions
    num_deconv_filters = [256, 128, 64]

    for i, (n_filt) in enumerate(num_deconv_filters):
        if n_filt <= 128:
            curr_block = conv_block
            block_type = "conv"
            transpose_block = conv_transpose_block
        else:
            curr_block = separable_conv_block
            block_type = "sepconv"
            transpose_block = sepconv_transpose_block

        x = curr_block(x,
                       filters=n_filt,
                       name=f'neck_{block_type}_{i}',
                       kernel_size=(3, 3),
                       padding='same',
                       use_bias=False,
                       add_activation=True,
                       add_batchnorm=True)
        x = transpose_block(x,
                            filters=n_filt,
                            name=f"neck_transpose_{block_type}_{i}",
                            kernel_size=(4, 4),
                            padding="same",
                            strides=(2, 2),
                            use_bias=False,
                            add_activation=True,
                            add_batchnorm=True)

    # Build the head which is composed of 2 consecutive convs
    bias_initializer = initializers.Constant(float(-np.log((1 - 0.1) / 0.1)))
    init_kernel = initializers.RandomNormal(stddev=0.001, seed=6)
    # In the legacy model there is 3 branches of 64 filters each one.
    # This could be merged in one branch
    x = conv_block(x, 3 * 64, (3, 3),
                   add_batchnorm=True,
                   use_bias=False,
                   add_activation=True,
                   padding="same",
                   name="head_conv_1",
                   kernel_initializer=init_kernel)
    # The output is built by #classes and box coordinates in xywh
    x = Conv2D(classes + 4,
               (1, 1),
               padding="same",
               use_bias=True,
               name="head_conv_2",
               bias_initializer=bias_initializer,
               kernel_initializer=init_kernel)(x)

    # Build the model
    return Model(inputs=base_model.input, outputs=x, name='centernet_base')
