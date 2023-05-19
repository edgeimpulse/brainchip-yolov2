#!/usr/bin/env python
# ******************************************************************************
# Copyright 2021 Brainchip Holdings Ltd.
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
Convtiny model definition for CWRU classification.
"""

from keras import Model
from keras.layers import Input, Softmax, Rescaling, Reshape

from cnn2snn import load_quantized_model

from ..layer_blocks import conv_block, separable_conv_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/convtiny/'


def convtiny_cwru():
    """ Instantiates a CNN for CWRU classification with input shape (32, 32, 1)
    and 10 classes.

    Returns:
        keras.Model: a Keras model for Convtiny/CWRU

    """
    img_input = Input(shape=(32, 32, 1), name="input")
    x = Rescaling(1, -127, name="rescaling")(img_input)
    x = conv_block(x,
                   filters=32,
                   name='conv_1',
                   kernel_size=(7, 7),
                   padding='same',
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    x = conv_block(x,
                   filters=32,
                   name='conv_2',
                   kernel_size=(7, 7),
                   padding='same',
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    x = separable_conv_block(x,
                             filters=512,
                             name='separable_1',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling='global_avg',
                             use_bias=False,
                             add_batchnorm=True,
                             add_activation=True)

    x = Reshape((1, 1, int(512)), name='reshape_1')(x)

    x = separable_conv_block(x,
                             filters=10,
                             name='predictions',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=False,
                             add_activation=False)
    x = Reshape((10,), name='reshape_2')(x)
    x = Softmax(name='act_softmax')(x)

    return Model(img_input, x, name='convtiny_cwru_32_10')


def convtiny_cwru_pretrained():
    """
    Helper method to retrieve a `convtiny_cwru` model that was trained on
    CWRU dataset.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'convtiny_cwru_iq8_wq2_aq4.h5'
    file_hash = '230b9e2e5901d5428fdba01f45f32ce261bcaabe4b5f861a58bed85e34f21f96'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
