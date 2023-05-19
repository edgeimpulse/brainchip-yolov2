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
DS-CNN model definition for KWS classification.
"""

from keras import Model
from keras.layers import (Input, Reshape, Activation, Flatten, Rescaling)

from cnn2snn import load_quantized_model, quantize

from ..layer_blocks import conv_block, separable_conv_block, dense_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/ds_cnn/'


def ds_cnn_kws(input_shape=(49, 10, 1),
               classes=33,
               include_top=True,
               weight_quantization=0,
               activ_quantization=0,
               input_weight_quantization=None,
               input_scaling=(255, 0)):
    """Instantiates a MobileNet-like model for the "Keyword Spotting" example.

    This model is based on the MobileNet architecture, mainly with fewer layers.
    The weights and activations are quantized such that it can be converted into
    an Akida model.

    This architecture is originated from https://arxiv.org/pdf/1711.07128.pdf
    and was created for the "Keyword Spotting" (KWS) or "Speech Commands"
    dataset.

    Args:
        input_shape (tuple): input shape tuple of the model
        classes (int): optional number of classes to classify words into, only
            be specified if `include_top` is True.
        include_top (bool): whether to include the fully-connected
            layer at the top of the model.
        weight_quantization (int): sets all weights in the model to have
            a particular quantization bitwidth except for the weights in the
            first layer.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization (int): sets all activations in the model to have a
            particular activation quantization bitwidth.

            * '0' implements floating point 32-bit activations.
            * '1' through '8' implements n-bit weights where n is from 2-8 bits.
        input_weight_quantization (int): sets weight quantization in the first
            layer. Defaults to weight_quantization value.

            * 'None' implements the same bitwidth as the other weights.
            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (255, 0). Note that following Akida convention,
            the scale factor is an integer used as a divider.

    Returns:
        keras.Model: a Keras model for MobileNet/KWS
    """
    # Overrides input weight quantization if None
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    if include_top and not classes:
        raise ValueError("If 'include_top' is True, 'classes' must be set.")

    img_input = Input(shape=input_shape, name="input")

    if input_scaling is None:
        x = img_input
    else:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    x = conv_block(x,
                   filters=64,
                   kernel_size=(5, 5),
                   padding='same',
                   strides=(2, 2),
                   use_bias=False,
                   name='conv_0',
                   add_batchnorm=True)

    x = separable_conv_block(x,
                             filters=64,
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             name='separable_1',
                             add_batchnorm=True)

    x = separable_conv_block(x,
                             filters=64,
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             name='separable_2',
                             add_batchnorm=True)

    x = separable_conv_block(x,
                             filters=64,
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             name='separable_3',
                             add_batchnorm=True)

    x = separable_conv_block(x,
                             filters=64,
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             name='separable_4',
                             pooling='global_avg',
                             add_batchnorm=True)

    shape = (1, 1, int(64))
    x = Reshape(shape, name='reshape_1')(x)

    if include_top:
        x = Flatten(name="flatten")(x)
        x = dense_block(x,
                        units=classes,
                        name='dense_5',
                        use_bias=True,
                        add_activation=False)
        act_function = 'softmax' if classes > 1 else 'sigmoid'
        x = Activation(act_function, name=f'act_{act_function}')(x)

    model = Model(img_input, x, name='ds_cnn_kws')

    if ((weight_quantization != 0) or (activ_quantization != 0) or
            (input_weight_quantization != 0)):
        # Converts a standard sequential Keras model to a CNN2SNN Keras
        # quantized model, compatible for Akida conversion.
        model = quantize(model=model,
                         weight_quantization=weight_quantization,
                         activ_quantization=activ_quantization,
                         input_weight_quantization=input_weight_quantization)
    return model


def ds_cnn_kws_pretrained():
    """
    Helper method to retrieve a `ds_cnn_kws` model that was trained on
    KWS dataset.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'ds_cnn_kws_iq8_wq4_aq4_laq1.h5'
    file_hash = '2ba6220bb9545857c99a327ec14d2d777420c7848cb6a9b17d87e5a01951fe6f'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
