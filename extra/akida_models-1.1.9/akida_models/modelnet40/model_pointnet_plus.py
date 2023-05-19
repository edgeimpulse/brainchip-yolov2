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
PointNet++ model definition for ModelNet40 classification.
"""

from keras import layers, Model, regularizers

from cnn2snn import quantize, load_quantized_model

from .pointnet_utils import get_reshape_factor
from ..layer_blocks import conv_block, dense_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/pointnet_plus/'


def pointnet_plus_modelnet40(selected_points=128,
                             features=3,
                             knn_points=64,
                             classes=40,
                             alpha=1.0,
                             weight_quantization=0,
                             activ_quantization=0):
    """ Instantiates a PointNet++ model for the ModelNet40 classification.

    This example implements the point cloud deep learning paper
    `PointNet (Qi et al., 2017) <https://arxiv.org/abs/1612.00593>`_. For a
    detailed introduction on PointNet see `this blog post
    <https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a>`_.

    PointNet++ is conceived as a repeated series of operations: sampling and
    grouping of points, followed by the trainable convnet itself. Those
    operations are then repeated at increased scale.
    Each of the selected points is taken as the centroid of the K-nearest
    neighbours. This defines a localized group.

    Args:
        selected_points (int, optional): the number of points to process per
            sample. Default is 128.
        features (int, optional): the number of features. Expected values are
            1 or 3. Default is 3.
        knn_points (int, optional): the number of points to include in each
            localised group. Must be a power of 2, and ideally an integer square
            (so 64, or 16 for a deliberately small network, or 256 for large).
            Default is 64.
        classes (int, optional): the number of classes for the classifier.
            Default is 40.
        alpha (float, optional): network filters multiplier. Default is 1.0.
        weight_quantization (int, optional): sets all weights in the model to
            have a particular quantization bitwidth except for the weights in
            the first layer. Defaults to 0.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization (int, optional): sets all activations in the model to
            have a particular activation quantization bitwidth. Defaults to 0.

            * '0' implements floating point 32-bit activations.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.

    Returns:
        keras.Model: a quantized Keras model for PointNet++/ModelNet40.
    """
    # Adapt input shape with preprocessing
    reshape_factor = get_reshape_factor(knn_points)
    input_shape = (selected_points * reshape_factor,
                   knn_points // reshape_factor, features)

    inputs = layers.Input(shape=input_shape, name="input")
    base_filter_num = round(32 * alpha)
    reg = regularizers.l1_l2(1e-7, 1e-7)

    # Block 1
    x = conv_block(inputs,
                   filters=base_filter_num,
                   name='block_1/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_1/conv_1/relu_1')(x)

    x = conv_block(x,
                   filters=base_filter_num,
                   name='block_1/conv_2',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_1/conv_2/relu_1')(x)
    x = layers.MaxPool2D(padding='same', name='max_pooling2d')(x)

    # Block 2
    x = conv_block(x,
                   filters=base_filter_num * 2,
                   name='block_2/conv_1',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_2/conv_1/relu_1')(x)
    x = conv_block(x,
                   filters=base_filter_num * 2,
                   name='block_2/conv_2',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_2/conv_2/relu_1')(x)
    if knn_points >= 8:
        x = layers.MaxPool2D(padding='same', name='max_pooling2d_1')(x)

    # Block 3
    x = conv_block(x,
                   filters=base_filter_num * 4,
                   name='block_3/conv_1',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_3/conv_1/relu_1')(x)
    x = conv_block(x,
                   filters=base_filter_num * 4,
                   name='block_3/conv_2',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_3/conv_2/relu_1')(x)
    if knn_points >= 32:
        x = layers.MaxPool2D(padding='same', name='max_pooling2d_2')(x)

    # Block 4
    x = conv_block(x,
                   filters=base_filter_num * 8,
                   name='block_4/conv_1',
                   kernel_size=(1, 1),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=False)
    x = layers.ReLU(max_value=6, activity_regularizer=reg, name='block_4/conv_1/relu_1')(x)
    if knn_points >= 128:
        x = layers.MaxPool2D(padding='same', name='max_pooling2d_3')(x)

    # Block 5
    x = conv_block(x,
                   filters=base_filter_num * 16,
                   name='block_5/conv_1',
                   kernel_size=(1, 1),
                   pooling='global_avg',
                   padding='same',
                   add_batchnorm=True)

    # Block 6
    x = dense_block(x,
                    units=base_filter_num * 16,
                    name='fc_1',
                    add_batchnorm=True)
    x = dense_block(x,
                    units=base_filter_num * 8,
                    name='fc_2',
                    add_batchnorm=True)
    x = layers.Dense(classes, activation=None, name="dense")(x)
    act_function = 'softmax' if classes > 1 else 'sigmoid'
    outputs = layers.Activation(act_function, name=f'act_{act_function}')(x)

    model = Model(inputs=inputs, outputs=outputs, name="pointnet_plus")

    if ((weight_quantization != 0) or (activ_quantization != 0)):
        # Converts a standard sequential Keras model to a CNN2SNN Keras
        # quantized model, compatible for Akida conversion.
        model = quantize(model=model,
                         weight_quantization=weight_quantization,
                         activ_quantization=activ_quantization,
                         input_weight_quantization=8)

    return model


def pointnet_plus_modelnet40_pretrained():
    """
    Helper method to retrieve a `pointnet_plus` model that was trained on
    ModelNet40 dataset.

    Returns:
        keras.Model: a Keras Model instance.
    """
    model_name = 'pointnet_plus_modelnet40_iq8_wq4_aq4.h5'
    file_hash = '9a49fcdf4742f0bfefb6e16c006d66d4064d9fc0ee30abc1bbd2341f5c5f8fec'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')

    return load_quantized_model(model_path)
