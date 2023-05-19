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
ViT model definition.
Inspired from https://github.com/faustomorales/vit-keras/blob/master/vit_keras/vit.py.
"""
import keras

from quantizeml.layers import AddPositionEmbs, ClassToken, Add, ExtractToken
from quantizeml.models import load_model

from .model_vit import CONFIG_TI, CONFIG_S, CONFIG_B
from ..imagenet.imagenet_utils import IMAGENET_MEAN, IMAGENET_STD
from ..layer_blocks import norm_to_layer, transformer_block
from ..utils import fetch_file

BASE_WEIGHT_PATH = 'http://data.brainchip.com/models/deit/'


def deit_imagenet(input_shape,
                  num_blocks,
                  hidden_size,
                  num_heads,
                  name,
                  mlp_dim,
                  patch_size=16,
                  classes=1000,
                  dropout=0.1,
                  include_top=True,
                  distilled=False,
                  norm='LN',
                  last_norm='LN',
                  softmax='softmax',
                  act="GeLU"):
    """Build a DeiT model.

    Args:
        input_shape (tuple): image shape tuple
        num_blocks (int): the number of transformer blocks to use.
        hidden_size (int): the number of filters to use
        num_heads (int): the number of transformer heads
        name (str): the model name
        mlp_dim (int): the number of dimensions for the MLP output in the transformers.
        patch_size (int, optional): the size of each patch (must fit evenly in image size). Defaults
            to 16.
        classes (int, optional): number of classes to classify images into, only to be specified if
            `include_top` is True. Defaults to 1000.
        dropout (float, optional): fraction of the units to drop for dense layers. Defaults to 0.1.
        include_top (bool, optional): whether to include the final classifier head. If False,
            the output will correspond to that of the transformer. Defaults to True.
        distilled (bool, optional): Build model append a distilled token. Defaults to False.
        norm (str, optional): string that values in ['LN', 'GN1', 'BN', 'LMN'] and that allows to
            choose from LayerNormalization, GroupNormalization(groups=1, ...), BatchNormalization
            or LayerMadNormalization layers respectively in the model. Defaults to 'LN'.
        last_norm (str, optional): string that values in ['LN', 'BN']
            and that allows to choose from LayerNormalization or
            BatchNormalization in the classifier network. Defaults to 'LN'.
        softmax (str, optional): string with values in ['softmax', 'softmax2']
            that allows to choose between softmax and softmax2 in MHA. Defaults
            to 'softmax'.
        act (str, optional): string that values in ['GeLU', 'ReLUx', 'swish'] and that allows to
            choose from GeLU, ReLUx or swish activation in MLP block. Defaults to 'GeLU'.
    """
    assert ((input_shape[0] % patch_size == 0) and
            (input_shape[1] % patch_size == 0)), "image size must be a multiple of patch_size"

    if last_norm not in ('LN', 'BN'):
        raise NotImplementedError("last_norm should be in ['LN', 'BN']"
                                  f"but received {norm}.")

    # Normalize image adding rescaling layer
    x = keras.layers.Input(shape=input_shape, name="input")
    scale = list(1.0 / 255 / std for std in IMAGENET_STD)
    offset = list(-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD))
    y = keras.layers.Rescaling(scale=scale, offset=offset, name="Rescale")(x)

    # Build model
    y = keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="Embedding",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        bias_initializer="zeros",
    )(y)
    y = keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    if distilled:
        y = ClassToken(name="DistToken")(y)
    y = ClassToken(name="ClassToken")(y)
    y = AddPositionEmbs(name="Transformer/PosEmbed")(y)
    for n in range(num_blocks):
        y, _ = transformer_block(
            y,
            num_heads=num_heads,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/EncoderBlock_{n}",
            norm=norm,
            softmax=softmax,
            mlp_act=act,
        )

    # Include classification head
    if include_top:
        yt = norm_to_layer(last_norm)(
            epsilon=1e-6, name="Transformer/EncoderNorm")(y)
        y = ExtractToken(token=0, name="ExtractToken")(yt)
        y = keras.layers.Dense(classes, name="Head")(y)
        if distilled:
            yd = ExtractToken(token=1, name="ExtractToken_Dist")(yt)
            yd = keras.layers.Dense(classes, name="DistHead")(yd)
            y = Add(name="Add", average=True)([y, yd])

    # Add distilled flag
    model = keras.models.Model(inputs=x, outputs=y, name=name)
    model.isdistilled = distilled
    return model


def deit_ti16(input_shape=(224, 224, 3),
              classes=1000,
              distilled=False,
              norm='LN',
              last_norm='LN',
              softmax='softmax',
              act='GeLU',
              include_top=True):
    """Build DeiT-Tiny.

    Args:
        input_shape (tuple, optional): input shape. Defaults to (224, 224, 3).
        classes (int, optional): number of classes. Defaults to 1000.
        distilled (bool, optional): build model appending a distilled token. Defaults to False.
        norm (str, optional): string that values in ['LN', 'GN1', 'BN', 'LMN'] and that allows to
            choose from LayerNormalization, GroupNormalization(groups=1, ...), BatchNormalization
            or LayerMadNormalization layers respectively in the model. Defaults to 'LN'.
        last_norm (str, optional): string that values in ['LN', 'BN']
            and that allows to choose from LayerNormalization or
            BatchNormalization in the classifier network. Defaults to 'LN'.
        softmax (str, optional): string with values in ['softmax', 'softmax2'] that allows to choose
            between softmax and softmax2 in attention block. Defaults to 'softmax'.
        act (str, optional): string that values in ['GeLU', 'ReLUx', 'swish'] and that allows to
            choose from GeLU, ReLUx or swish activation inside MLP. Defaults to 'GeLU'.
        include_top (bool, optional): whether to include the final classifier network.
            Defaults to True.

    Returns:
        keras.Model: the requested model
    """
    return deit_imagenet(
        name="deit-tiny",
        input_shape=input_shape,
        classes=classes,
        distilled=distilled,
        norm=norm,
        last_norm=last_norm,
        act=act,
        softmax=softmax,
        include_top=include_top,
        **CONFIG_TI,
    )


def bc_deit_ti16(input_shape=(224, 224, 3), classes=1000, distilled=False, include_top=True,
                 num_blocks=12):
    """Build DeiT-Tiny, changing all LN by LMN, using softmax2 and ReLU8.

    Args:
        input_shape (tuple, optional): input shape. Defaults to (224, 224, 3).
        classes (int, optional): number of classes. Defaults to 1000.
        distilled (bool, optional): build model appending a distilled token. Defaults to False.
        include_top (bool, optional): whether to include the final classifier network.
            Defaults to True.
        num_blocks (int, optional): the number of transformer blocks to use. Defaults to 12.

    Returns:
        keras.Model: the requested model
    """
    config_ti = CONFIG_TI.copy()
    config_ti["num_blocks"] = num_blocks

    return deit_imagenet(
        name="deit-tiny",
        input_shape=input_shape,
        classes=classes,
        distilled=distilled,
        norm="LMN",
        last_norm="BN",
        softmax="softmax2",
        act="ReLU8",
        include_top=include_top,
        **config_ti,
    )


def bc_deit_dist_ti16_imagenet_pretrained():
    """ Helper method to retrieve a `bc_deit_dist_ti16` model that was trained on ImageNet dataset.

    Returns:
        keras.Model, dict: a Keras Model instance and the quantization configuration as a JSON
        formatted dict.
    """
    model_name = 'bc_deit_dist_ti16_224_quant_config_4bit.h5'
    file_hash = '0d8b1c46e215be149974e6116e2e88a50a9c6d2186f7131f4fb0b3f23aaa2aac'
    model_path = fetch_file(BASE_WEIGHT_PATH + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')
    return load_model(model_path)


def deit_s16(input_shape=(224, 224, 3), classes=1000, distilled=False, include_top=True):
    """Build DeiT-Small.

    Args:
        input_shape (tuple, optional): input shape. Defaults to (224, 224, 3).
        classes (int, optional): number of classes. Defaults to 1000.
        distilled (bool, optional): build model appending a distilled token. Defaults to False.
        include_top (bool, optional): whether to include the final classifier network.
            Defaults to True.

    Returns:
        keras.Model: the requested model
    """
    return deit_imagenet(
        name="deit-small",
        input_shape=input_shape,
        classes=classes,
        distilled=distilled,
        include_top=include_top,
        **CONFIG_S,
    )


def deit_b16(input_shape=(224, 224, 3), classes=1000, distilled=False, include_top=True):
    """Build DeiT-B16.

    Args:
        input_shape (tuple, optional): input shape. Defaults to (224, 224, 3).
        classes (int, optional): number of classes. Defaults to 1000.
        distilled (bool, optional): build model appending a distilled token. Defaults to False.
        include_top (bool, optional): whether to include the final classifier network.
            Defaults to True.

    Returns:
        keras.Model: the requested model
    """
    return deit_imagenet(
        name="deit-base",
        input_shape=input_shape,
        classes=classes,
        distilled=distilled,
        include_top=include_top,
        **CONFIG_B,
    )
