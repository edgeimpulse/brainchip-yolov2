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
Provides utilities to preprocess CIFAR-10 images.

"""

import tensorflow as tf


def resize_image(image_buffer, output_height=224, output_width=224):
    """Resize the given image.

  Args:
    image_buffer (tf.Tensor): scalar string Tensor representing the raw JPEG image buffer.
    output_height (int): the height of the image after preprocessing.
    output_width (int): the width of the image after preprocessing.

  Returns:
        tf.Tensor: resized image in float32 format.
    """

    image = tf.cast(image_buffer, tf.float32)
    image = tf.image.resize(image, (output_height, output_width))
    return image
