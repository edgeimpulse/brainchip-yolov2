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
Convtiny model definition for DVS handy classification.
"""

from keras import Model
from keras.layers import Input, Reshape, Dropout, Activation

from cnn2snn import load_quantized_model, quantize

from ..layer_blocks import conv_block, separable_conv_block, dense_block
from ..utils import fetch_file

# Locally fixed config options
# The number of neurons in the penultimate dense layer
# This layer has binary output spikes, and could be a bottleneck
# if care isn't taken to ensure enough info capacity
NUM_SPIKING_NEURONS = 256

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/convtiny/'


def convtiny_dvs_handy(input_shape=(120, 160, 2),
                       classes=9,
                       weight_quantization=0,
                       activ_quantization=0):
    """Instantiates a CNN for "Brainchip dvs_handy" example.

    Args:
        input_shape (tuple): input shape tuple of the model
        classes (int, optional): number of classes to classify images into. Defaults to 9.
        weight_quantization (int): sets all weights in the model to have
            a particular quantization bitwidth except for the weights in the
            first layer.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization: sets all activations in the model to have a
            particular activation quantization bitwidth.

            * '0' implements floating point 32-bit activations.
            * '1' through '8' implements n-bit weights where n is from 1-8 bits.

    Returns:
        keras.Model: a Keras convolutional model for DVS Gesture.
    """

    img_input = Input(input_shape, name="input")

    x = conv_block(img_input,
                   filters=16,
                   kernel_size=(3, 3),
                   name='conv_0',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='max',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    x = conv_block(x,
                   filters=32,
                   kernel_size=(3, 3),
                   name='conv_1',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='max',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    x = conv_block(x,
                   filters=64,
                   kernel_size=(3, 3),
                   name='conv_2',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='max',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    x = conv_block(x,
                   filters=128,
                   kernel_size=(3, 3),
                   name='conv_3',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='max',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    x = conv_block(x,
                   filters=256,
                   kernel_size=(3, 3),
                   name='conv_4',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='max',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    x = conv_block(x,
                   filters=512,
                   kernel_size=(3, 3),
                   name='conv_5',
                   use_bias=False,
                   add_batchnorm=True,
                   padding='same',
                   pooling='global_avg',
                   pool_size=(2, 2),
                   add_activation=True,
                   strides=(1, 1))

    bm_outshape = (1, 1, 512)

    x = Reshape(bm_outshape, name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)

    x = separable_conv_block(x,
                             filters=NUM_SPIKING_NEURONS,
                             kernel_size=(3, 3),
                             use_bias=False,
                             padding='same',
                             name='spiking_layer',
                             add_batchnorm=True,
                             add_activation=True,
                             pooling=None)

    x = dense_block(x,
                    units=classes,
                    name='dense',
                    add_batchnorm=False,
                    add_activation=False,
                    use_bias=False)

    act_function = 'softmax' if classes > 1 else 'sigmoid'
    x = Activation(act_function, name=f'act_{act_function}')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(inputs=img_input, outputs=x, name='dvs_network')

    if ((weight_quantization != 0) or (activ_quantization != 0)):
        return quantize(model, weight_quantization, activ_quantization)
    return model


def convtiny_handy_samsung_pretrained():
    """ Helper method to retrieve a `convtiny_dvs_handy` model that was trained
    on samsung_handy dataset.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'convtiny_dvs_handy_samsung_iq4_wq4_aq4.h5'
    file_hash = 'ac5dbf1420fbedc402da4394bb22cf94ff5cff73adb428cca741d6550f663c71'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
