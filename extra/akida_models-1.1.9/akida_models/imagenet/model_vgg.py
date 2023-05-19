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
VGG model definition for ImageNet classification.
"""

from keras import Model, regularizers
from keras.layers import (Input, Activation, Dropout, Flatten, Rescaling)

from cnn2snn import quantize, load_quantized_model

from ..layer_blocks import conv_block, dense_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/vgg/'


def vgg_imagenet(input_shape=(224, 224, 3),
                 classes=1000,
                 include_top=True,
                 pooling=None,
                 weight_quantization=0,
                 activ_quantization=0,
                 input_weight_quantization=None,
                 input_scaling=(128, -1)):
    """Instantiates a VGG11 architecture with reduced number of filters in
    convolutional layers (i.e. a quarter of the filters of the original
    implementation of https://arxiv.org/pdf/1409.1556.pdf).

    Args:
        input_shape (tuple, optional): input shape tuple. Defaults to (224, 224,
            3).
        classes (int, optional): optional number of classes to classify images
            into. Defaults to 1000.
        include_top (bool, optional): whether to include the classification
            layers at the top of the model. Defaults to True.
        pooling (str, optional): Optional pooling mode for feature extraction
            when `include_top` is `False`. Defaults to None.

            * `None` means that the output of the model will be the 4D tensor
              output of the last convolutional block.
            * `avg` means that global average pooling will be applied to the
              output of the last convolutional block, and thus the output of the
              model will be a 2D tensor.
        weight_quantization (int, optional): sets all weights in the model to
            have a particular quantization bitwidth except for the weights in
            the first layer. Defaults to 0.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization (int, optional): sets all activations in the model to
            have a particular activation quantization bitwidth. Defaults to 0.

            * '0' implements floating point 32-bit activations.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_weight_quantization(int, optional): sets weight quantization in
            the first layer. Defaults to weight_quantization value.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (128, -1). Note that following Akida convention,
            the scale factor is an integer used as a divider.

    Returns:
        keras.Model: a Keras model for VGG/ImageNet

    """
    # check if overrides have been provided and override
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    # Define weight regularization
    weight_regularizer = regularizers.l2(4e-5)

    img_input = Input(shape=input_shape, name="input")

    if input_scaling is None:
        x = img_input
    else:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    # Block 1
    x = conv_block(x,
                   filters=16,
                   name='block_1/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    # Block 2
    x = conv_block(x,
                   filters=32,
                   name='block_2/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    # Block 3
    x = conv_block(x,
                   filters=64,
                   name='block_3/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True)

    x = conv_block(x,
                   filters=64,
                   name='block_3/conv_2',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    # Block 4
    x = conv_block(x,
                   filters=128,
                   name='block_4/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True)

    x = conv_block(x,
                   filters=128,
                   name='block_4/conv_2',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    # Block 5
    x = conv_block(x,
                   filters=128,
                   name='block_5/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True)

    layer_pooling = 'global_avg' if pooling == 'avg' else 'max'
    x = conv_block(x,
                   filters=128,
                   name='block_5/conv_2',
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_regularizer=weight_regularizer,
                   add_batchnorm=True,
                   add_activation=True,
                   pooling=layer_pooling,
                   pool_size=(2, 2))

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = dense_block(x,
                        units=4096,
                        name='fc_1',
                        add_batchnorm=True,
                        add_activation=True)
        x = Dropout(0.5, name='dropout_1')(x)
        x = dense_block(x,
                        units=4096,
                        name='fc_2',
                        add_batchnorm=True,
                        add_activation=True)
        x = Dropout(0.5, name='dropout_2')(x)
        x = dense_block(x,
                        units=classes,
                        name='predictions',
                        add_batchnorm=False,
                        add_activation=False)

        act_function = 'softmax' if classes > 1 else 'sigmoid'
        x = Activation(act_function, name=f'act_{act_function}')(x)

    # Create model
    model = Model(img_input, x, name='vgg11_%s_%s' % (input_shape[0], classes))

    if ((weight_quantization != 0) or (activ_quantization != 0) or
            (input_weight_quantization != 0)):
        return quantize(model, weight_quantization, activ_quantization,
                        input_weight_quantization)
    return model


def vgg_imagenet_pretrained():
    """
    Helper method to retrieve a `vgg_imagenet` model that was trained on
    ImageNet dataset.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'vgg11_imagenet_224_iq8_wq4_aq4.h5'
    file_hash = '40fe5e7fdf083604d3bbe8eba314a832d1dc1f218a743bef244ff6ce2f3c9bfe'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
