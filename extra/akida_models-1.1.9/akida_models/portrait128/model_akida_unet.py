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
Akida U-Net model definition for semantic segmentation.
"""

from keras import Model, regularizers
from keras.layers import Dropout, Activation

from ..layer_blocks import sepconv_transpose_block, dense_block
from ..imagenet.model_akidanet import akidanet_imagenet


def akida_unet_portrait128(input_shape=(128, 128, 3),
                           alpha=0.5,
                           input_scaling=(128, -1)):
    """Instantiates an Akida U-Net architecture.

    It is composed of an AkidaNet-ImageNet encoder followed by a succession of Conv2DTranspose
    layers for the decoder part.
    It does not contain any skip connection (concatenation) between the encoder and the decoder
    branches.

    Args:
        input_shape (tuple, optional): input shape tuple. Defaults to (128, 128, 3).
        alpha (float, optional): controls the width (number of filters) of the model. Defaults to
            0.5.
        input_scaling (tuple, optional): scale factor and offset to apply to inputs. Defaults to
            (128, -1). Note that following Akida convention, the scale factor is a number used as a
            divider.

    Returns:
        keras.Model: a Keras Model instance.
    """
    # Define weight regularization, will apply to pointwise weights of sepconv transposed layers
    weight_regularizer = regularizers.l2(4e-5)

    # Create an AkidaNet network without top layers
    encoder = akidanet_imagenet(input_shape=input_shape,
                                alpha=alpha,
                                include_top=False,
                                input_scaling=input_scaling)

    # Add the decoder layers
    x = encoder.layers[-1].output
    x = sepconv_transpose_block(x,
                                filters=int(512 * alpha),
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                kernel_initializer='he_normal',
                                add_batchnorm=True,
                                name='sepconv_t_0',
                                pointwise_regularizer=weight_regularizer)
    x = Dropout(0.5)(x)
    x = sepconv_transpose_block(x,
                                filters=int(256 * alpha),
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                kernel_initializer='he_normal',
                                add_batchnorm=True,
                                name='sepconv_t_1',
                                pointwise_regularizer=weight_regularizer)
    x = Dropout(0.5)(x)
    x = sepconv_transpose_block(x,
                                filters=int(128 * alpha),
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                kernel_initializer='he_normal',
                                add_batchnorm=True,
                                name='sepconv_t_2',
                                pointwise_regularizer=weight_regularizer)
    x = Dropout(0.5)(x)
    x = sepconv_transpose_block(x,
                                filters=int(64 * alpha),
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                kernel_initializer='he_normal',
                                add_batchnorm=True,
                                name='sepconv_t_3',
                                pointwise_regularizer=weight_regularizer)
    x = Dropout(0.5)(x)
    x = sepconv_transpose_block(x,
                                filters=int(32 * alpha),
                                kernel_size=(3, 3),
                                strides=2,
                                padding='same',
                                kernel_initializer='he_normal',
                                add_batchnorm=True,
                                name='sepconv_t_4',
                                pointwise_regularizer=weight_regularizer)
    x = Dropout(0.5)(x)
    x = dense_block(x, units=1, add_activation=False, name='head')
    x = Activation('sigmoid', name="sigmoid_act")(x)

    # Build the whole model: encoder followed by decoder
    return Model(inputs=encoder.input, outputs=x, name='akida_unet')
