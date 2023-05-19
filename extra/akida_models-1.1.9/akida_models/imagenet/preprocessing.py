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
Preprocessing tools for ImageNet dataset.
"""

import numpy as np
import keras
import tensorflow as tf
import tensorflow_addons as tfa

from akida_models.imagenet.imagenet_labels2names import imagenet_labels


class RandomColorJitter(keras.layers.Layer):
    """RandomColorJitter class.

    Randomly adds color jitter to an image. Color jitter means to add random brightness, contrast,
    saturation, and hue to an image. There is a 80% chance that an image will be randomly
    color-jittered. Taken on https://keras.io/examples/vision/barlow_twins/

    Args:
        proba(float, optional): Probability of applying the color jitter. Defaults to 0.8.
    """

    def __init__(self, *args, proba=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.proba = proba

    @tf.function
    def call(self, image):
        def _color_jitter(image):
            image = tf.image.random_brightness(image, 0.8)
            image = tf.image.random_contrast(image, 0.4, 1.6)
            image = tf.image.random_saturation(image, 0.4, 1.6)
            image = tf.image.random_hue(image, 0.2)
            return image
        return tf.cond(
            tf.random.uniform([]) < self.proba, lambda: _color_jitter(image), lambda: image)


class ThreeAugment(keras.layers.Layer):
    """Define a simple data augmentation pipeline of three augmentations, following the explaining
    in the paper: https://arxiv.org/abs/2204.07118. For that, this augmentation choses one of:
        - GrayScale: This favors color invariance and give more focus on shapes.
        - Solarization: This adds strong noise on the colour to be more robust to the variation
          of colour intensity and so focus more on shape
        - Gaussian Blur: In order to slightly alter details in the image.
    """
    @tf.function
    def call(self, image):
        def _to_gray():
            return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

        def _solarize():
            # Taken of: https://keras.io/examples/vision/barlow_twins/
            return tf.where(image < 10, image, 255 - image)

        def _gaussian_blur():
            # Taken of: https://keras.io/examples/vision/barlow_twins/
            s = np.random.random()
            return tfa.image.gaussian_filter2d(image=image, sigma=s)

        proba = tf.random.uniform([])
        cases = [(proba < 1 / 3, _to_gray), (proba < 2 / 3, _solarize)]
        return tf.case(cases, default=_gaussian_blur, name='3-Augment', exclusive=False)


DATA_AUGMENTATION = keras.Sequential([RandomColorJitter(), ThreeAugment()])


@tf.function
def preprocess_image(image, image_size, training=False, data_aug=None):
    """ ImageNet data preprocessing.

    Preprocessing includes cropping, and resizing for both training and
    validation images. Training preprocessing introduces some random distortion
    of the image to improve accuracy.

    Args:
        image (tf.Tensor): input image as a 3-D tensor
        image_size (int): desired image size
        training (bool, optional): True for training preprocessing, False for
            validation and inference. Defaults to False.
        data_aug (keras.Sequential, optional): data augmentation. Defaults to None.

    Returns:
        :obj:`tensorflow.Tensor`: preprocessed image
    """
    shape = tf.shape(image)

    if training:
        # For training: crop, flip and resize
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            shape,
            tf.zeros([0, 0, 4], tf.float32),  # force using whole image
            use_image_if_no_bounding_boxes=True,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100)

        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.image.resize(image, [image_size, image_size])

        # Make all data augmentation after resize to decrease computational cost
        image = tf.image.random_flip_left_right(image)
        if data_aug is not None:
            image = data_aug(image)
    else:
        # For validation/inference: aspect preserving resize and central crop
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)

        # Scale image before cropping, keeping aspect ratio
        resize_min = np.round(image_size * 1.143).astype(np.float32)
        scale_ratio = resize_min / tf.minimum(height, width)

        # Convert back to int for TF ops
        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)

        image = tf.image.resize(image, [new_height, new_width])

        # Second: central crop to desired image_size
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)

    return tf.cast(image, tf.float32)


def index_to_label(index):
    """ Function to get an ImageNet label from an index.

    Args:
        index: between 0 and 999

    Returns:
        str: a string of comma separated labels
    """
    return imagenet_labels[index]
