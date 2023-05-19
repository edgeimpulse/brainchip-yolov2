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
GXNOR model definition for MNIST classification.
"""

from keras import Model
from keras.layers import Input, Flatten, Rescaling

from cnn2snn import quantize, load_quantized_model

from ..layer_blocks import conv_block, dense_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/gxnor/'


def gxnor_mnist(weight_quantization=0,
                activ_quantization=0,
                input_weight_quantization=None):
    """ Instantiates a Keras GXNOR model with an additional dense layer to make
    better classification.

    The paper describing the original model can be found `here
    <https://www.sciencedirect.com/science/article/pii/S0893608018300108>`_.

    Args:
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

    Returns:
        keras.Model: a Keras model for GXNOR/MNIST
    """

    # check if overrides have been provided and override
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    img_input = Input(shape=(28, 28, 1), name="input")
    x = Rescaling(1. / 255, name="rescaling")(img_input)

    # Block 1
    x = conv_block(x,
                   filters=32,
                   name='block_1/conv_1',
                   kernel_size=(5, 5),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=True,
                   pooling='max',
                   pool_size=(2, 2))

    # Block 2
    x = conv_block(x,
                   filters=64,
                   name='block_2/conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   add_batchnorm=True,
                   add_activation=True,
                   strides=2,
                   pool_size=(2, 2))

    # Classification block
    x = Flatten(name='flatten')(x)
    x = dense_block(x,
                    units=512,
                    name='fc_1',
                    add_batchnorm=True,
                    add_activation=True)
    x = dense_block(x,
                    units=10,
                    name='predictions',
                    add_batchnorm=True,
                    add_activation=False)

    # Create model
    model = Model(img_input, x, name='gxnor_mnist')

    if ((weight_quantization != 0) or (activ_quantization != 0) or
            (input_weight_quantization != 0)):
        return quantize(model, weight_quantization, activ_quantization,
                        input_weight_quantization)
    return model


def gxnor_mnist_pretrained():
    """ Helper method to retrieve a `gxnor_mnist` model that was trained on
    MNIST dataset.

    This model was trained with the distillation knowledge method, using
    the EfficientNet model from `this repository
    <https://github.com/EscVM/Efficient-CapsNet>`_ and the `Distiller` class from
    the knowledge distillation toolkit (`akida_models.distiller`).

    The float training was done for 30 epochs with a learning rate of 1e-4
    After that we gradually quantize the model from:
    8-4-4 --> 4-4-4 --> 4-4-2 --> 2-2-2 --> 2-2-1 tuning the model at each step
    with the same distillation training method for 5 epochs and a learning rate
    of 5e-5.

    Returns:
        keras.Model: a Keras Model instance.

    """
    model_name = 'gxnor_mnist_iq2_wq2_aq1.h5'
    file_hash = '836d4d32bf6a25620782f783419a1eae0c90460f3c291e2c5a8ed4a6cf36b035'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_quantized_model(model_path)
