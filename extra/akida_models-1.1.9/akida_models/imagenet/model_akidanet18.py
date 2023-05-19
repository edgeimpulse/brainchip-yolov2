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
AkidaNet18 model definition for ImageNet classification.

AkidaNet18 architecture is inspired both from AkidaNet and ResNet18: same depth and dimensions than
ResNet18 but without skip connections and using SeparableConvolution layers.

"""
from keras import Model, regularizers
from keras.layers import Input, Rescaling

from cnn2snn import quantize, load_quantized_model

from .imagenet_utils import obtain_input_shape
from ..layer_blocks import conv_block, separable_conv_block, dense_block
from ..utils import fetch_file


BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/akidanet18/'


def akidanet18_imagenet(input_shape=None,
                        include_top=True,
                        pooling=None,
                        classes=1000,
                        depths=(4, 4, 4, 4),
                        dimensions=(64, 128, 256, 512),
                        weight_quantization=0,
                        activ_quantization=0,
                        input_weight_quantization=None,
                        input_scaling=(128, -1)):
    """Instantiates the AkidaNet18 architecture.

    Args:
        input_shape (tuple, optional): shape tuple. Defaults to None.
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the model. Defaults to True.
        pooling (str, optional): optional pooling mode for feature extraction
            when `include_top` is `False`.
            Defaults to None.

            * `None` means that the output of the model will be the 4D tensor
              output of the last convolutional block.
            * `avg` means that global average pooling will be applied to the
              output of the last convolutional block, and thus the output of the
              model will be a 2D tensor.
        classes (int, optional): optional number of classes to classify images
            into, only to be specified if `include_top` is `True`. Defaults to 1000.
        depth (tuple, optional): number of layers in each stages of the model. The length of the
            tuple defines the number of stages. Defaults to (4, 4, 4, 4).
        dimensions (tuple, optional): number of filters in each stage on the model. The length of
            the tuple must be equal to the length of the `depth` tuple. Defaults to
            (64, 128, 256, 512).
        weight_quantization (int, optional): sets all weights in the model to have
            a particular quantization bitwidth except for the weights in the
            first layer.
            Defaults to 0.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization (int, optional): sets all activations in the model to have a
            particular activation quantization bitwidth.
            Defaults to 0.

            * '0' implements floating point 32-bit activations.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_weight_quantization (int, optional): sets weight quantization in the first layer.
            Defaults to weight_quantization value.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (128, -1). Note that following Akida convention,
            the scale factor is an integer used as a divider.

    Returns:
        keras.Model: a Keras model for AkidaNet/ImageNet.

    Raises:
        ValueError: in case of invalid input shape or mismatching `depth` and `dimensions`.
    """
    # Check if overrides have been provided and override
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    # Sanity checks
    stages = len(depths)
    if len(dimensions) != stages:
        raise ValueError(f"'depth' and 'dimensions' must be of the same length, received: {depths} "
                         f"and {dimensions}.")

    # Define weight regularization, will apply to the convolutional layers and
    # to all pointwise weights of separable convolutional layers.
    weight_regularizer = regularizers.l2(4e-5)

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        rows = input_shape[0]
        cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = obtain_input_shape(input_shape,
                                     default_size=default_size,
                                     min_size=32,
                                     include_top=include_top)

    img_input = Input(shape=input_shape, name="input")

    if input_scaling is None:
        x = img_input
    else:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    # ConvNext stem layer: 4x4 kernel with stride 4
    x = conv_block(x,
                   filters=int(dimensions[0]),
                   name='convnext_stem',
                   kernel_size=(4, 4),
                   padding='same',
                   use_bias=False,
                   strides=4,
                   add_batchnorm=True,
                   add_activation=True,
                   kernel_regularizer=weight_regularizer)

    # Define the stages
    for stage in range(stages):
        # Like for AkidaNet, early layers (first 2 stages) are defined as standard Convolutional and
        # next layers are SeparableConvolutional layers
        if stage < 2:
            current_block = conv_block
        else:
            current_block = separable_conv_block

        strides = 2 if stage > 0 else 1
        for i in range(depths[stage]):
            # First layer in stage comes with strides 2 except in first stage where strides is
            # handled by the previous stem
            strides = 2 if i == 0 and stage > 0 else 1

            # Handle final pooling in last layer of last stage
            if stage == stages - 1 and i == depths[stage] - 1:
                pool = 'global_avg' if include_top or pooling == 'avg' else None
            else:
                pool = None

            x = current_block(x,
                              filters=int(dimensions[stage]),
                              name=f'stage_{stage}/conv_{i}',
                              kernel_size=(3, 3),
                              strides=strides,
                              padding='same',
                              use_bias=False,
                              pooling=pool,
                              add_batchnorm=True,
                              add_activation=True,
                              kernel_regularizer=weight_regularizer)

    # Classification layer
    if include_top:
        x = dense_block(x,
                        classes,
                        add_batchnorm=False,
                        add_activation=False,
                        kernel_initializer="he_normal",
                        name='classifier',
                        kernel_regularizer=weight_regularizer)

    # Create model
    model = Model(img_input, x, name='akidanet18_%s_%s' % (input_shape[0], classes))

    if ((weight_quantization != 0) or (activ_quantization != 0) or
            (input_weight_quantization != 0)):
        return quantize(model, weight_quantization, activ_quantization,
                        input_weight_quantization)
    return model


def akidanet18_imagenet_pretrained():
    """
    Helper method to retrieve an `akidanet18_imagenet` model that was trained on ImageNet dataset.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'akidanet18_imagenet_224_iq8_wq4_aq4.h5'
    file_hash = 'd3a0a3f37b5d3a5998981705e772b40670fdab0dd68e5894dc8af717bbaae136'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
