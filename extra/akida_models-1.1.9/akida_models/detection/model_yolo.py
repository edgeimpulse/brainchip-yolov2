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
YOLO model definition for detection.
"""

import pickle

import numpy as np

from keras import Model

from cnn2snn import load_quantized_model, quantize

from ..layer_blocks import yolo_head_block
from ..imagenet.model_akidanet import akidanet_imagenet
from ..utils import fetch_file


def yolo_base(input_shape=(224, 224, 3),
              classes=1,
              nb_box=5,
              alpha=1.0,
              weight_quantization=0,
              activ_quantization=0,
              input_weight_quantization=None,
              input_scaling=(127.5, -1)):
    """ Instantiates the YOLOv2 architecture.

    Args:
        input_shape (tuple): input shape tuple
        classes (int): number of classes to classify images into
        nb_box (int): number of anchors boxes to use
        alpha (float): controls the width of the model
        weight_quantization (int): sets all weights in the model to have
            a particular quantization bitwidth except for the weights in the
            first layer.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization: sets all activations in the model to have a
            particular activation quantization bitwidth.

            * '0' implements floating point 32-bit activations.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_weight_quantization: sets weight quantization in the first layer.
            Defaults to weight_quantization value.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (127.5, -1). Note that following Akida
            convention, the scale factor is a number used as a divider.

    Returns:
        keras.Model: a Keras Model instance.

    """
    # Overrides input weight quantization if None
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    # Create an AkidaNet network without top layers
    base_model = akidanet_imagenet(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weight_quantization=weight_quantization,
        activ_quantization=activ_quantization,
        input_weight_quantization=input_weight_quantization,
        input_scaling=input_scaling)

    # Add YOLO top layers to the base model
    input_shape = base_model.input_shape
    x = yolo_head_block(base_model.layers[-1].output, num_boxes=nb_box, classes=classes)
    model = Model(inputs=base_model.input, outputs=x, name='yolo_base')

    # Initialize detection layer weights
    layer = model.get_layer("detection_layer")
    detection_weights = layer.get_weights()

    mu, sigma = 0, 0.1

    grid_size = model.output_shape[1:3]
    grid_area = grid_size[0] * grid_size[1]

    dw_kernel = np.random.normal(mu, sigma,
                                 size=detection_weights[0].shape) / grid_area
    pw_kernel = np.random.normal(mu, sigma,
                                 size=detection_weights[1].shape) / grid_area
    bias = np.random.normal(mu, sigma,
                            size=detection_weights[2].shape) / grid_area

    layer.set_weights([dw_kernel, pw_kernel, bias])

    if ((weight_quantization != 0) or (activ_quantization != 0) or
            (input_weight_quantization != 0)):
        return quantize(model, weight_quantization, activ_quantization,
                        input_weight_quantization)
    return model


def yolo_widerface_pretrained():
    """
    Helper method to retrieve a `yolo_base` model that was trained on WiderFace
    dataset and the anchors that are needed to interpet the model output.

    Returns:
        keras.Model, list: a Keras Model instance and a list of anchors.

    """
    anchors_name = 'widerface_anchors.pkl'
    anchors_path = fetch_file(
        'http://data.brainchip.com/dataset-mirror/widerface/' + anchors_name,
        fname=anchors_name,
        file_hash='325f92336a310d83fed71765436ee343bf3e39cbc12fd099d30677761aee9376',
        cache_subdir='datasets/widerface')
    with open(anchors_path, 'rb') as handle:
        anchors = pickle.load(handle)

    model_name = 'yolo_akidanet_widerface_iq8_wq4_aq4.h5'
    file_hash = 'd55744cbfbbe1131aa26015f38d99f1df7026347bae8f66683259e366e7b6e03'
    model_path = fetch_file('http://data.brainchip.com/models/yolo/' + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')

    return load_quantized_model(model_path), anchors


def yolo_voc_pretrained():
    """
    Helper method to retrieve a `yolo_base` model that was trained on PASCAL
    VOC2012 dataset for 'person' and 'car' classes only, and the anchors that
    are needed to interpet the model output.

    Returns:
        keras.Model, list: a Keras Model instance and a list of anchors.

    """
    anchors_name = 'voc_anchors.pkl'
    anchors_path = fetch_file(
        'http://data.brainchip.com/dataset-mirror/voc/' + anchors_name,
        fname=anchors_name,
        file_hash='b1fe1ed12691e100646cf52b1320f05abd17b2f546d3e12cdee87758cc9ed0ba',
        cache_subdir='datasets/voc')
    with open(anchors_path, 'rb') as handle:
        anchors = pickle.load(handle)

    model_name = 'yolo_akidanet_voc_iq8_wq4_aq4.h5'
    file_hash = 'e65b0b6bd4b08c2796c3bbea89343748195e29e240ce28e70489c53d06ca69d9'
    model_path = fetch_file('http://data.brainchip.com/models/yolo/' + model_name,
                            fname=model_name,
                            file_hash=file_hash,
                            cache_subdir='models')

    return load_quantized_model(model_path), anchors
