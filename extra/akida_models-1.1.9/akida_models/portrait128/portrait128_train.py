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
Portrait128 training script.
"""

import os
import argparse

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import BinaryIoU

from cnn2snn import load_quantized_model

from akida_models.param_scheduler import get_cosine_lr_scheduler

from ..training import (get_training_parser, compile_model, evaluate_model, print_history_stats,
                        RestoreBest)


def get_data(path, batch_size):
    """ Loads Portrait128 data.

    Args:
        path (str): path to npy data
        batch_size (int): the batch size

    Returns:
        tuple: train generator, validation generator, steps per epoch and validation step
    """
    # Load the dataset
    x_train = np.load(os.path.join(path, "img_uint8.npy"))
    y_train = np.load(os.path.join(path, "msk_uint8.npy")) / 255

    # Data generator for training and validation
    data_gen_args = dict(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         validation_split=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    train_image_generator = image_datagen.flow(x_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               subset='training',
                                               seed=seed)

    train_mask_generator = mask_datagen.flow(y_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             subset='training',
                                             seed=seed)

    val_image_generator = image_datagen.flow(x_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             subset='validation',
                                             seed=seed)

    val_mask_generator = mask_datagen.flow(y_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           subset='validation',
                                           seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    return (train_generator, val_generator, train_image_generator.n // batch_size,
            val_image_generator.n // batch_size)


def train_model(model, train_gen, steps_per_epoch, val_gen, val_steps, epochs, learning_rate):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        train_gen (keras.ImageDataGenerator): train data generator
        steps_per_epoch (int): training steps
        val_gen (keras.ImageDataGenerator): validation data generator
        val_steps (int): validation steps
        epochs (int): the number of epochs
        learning_rate (float): the learning rate
    """
    # Define learning rate scheduler
    callbacks = [get_cosine_lr_scheduler(learning_rate, epochs * steps_per_epoch, True)]

    # Model checkpoints (save best model and retrieve it when training is complete)
    restore_model = RestoreBest(model, monitor="val_binary_io_u")
    callbacks.append(restore_model)

    history = model.fit(train_gen,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps,
                        validation_data=val_gen,
                        callbacks=callbacks)
    print_history_stats(history)


def main():
    """ Entry point for script and CLI usage.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("-d", "--data", type=str,
                               default="/hdd/datasets/Portrait128/data/",
                               help="Path to the Portrait128 data.")

    parsers = get_training_parser(batch_size=32, tune=True, global_parser=global_parser)
    args = parsers[0].parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    learning_rate = 3e-5
    if args.action == "tune":
        learning_rate /= 10

    compile_model(model, learning_rate=learning_rate, loss='binary_crossentropy',
                  metrics=[BinaryIoU(), 'accuracy'])

    # Load data
    train_gen, val_gen, steps_per_epoch, val_steps = get_data(args.data, args.batch_size)

    # Train model
    if args.action in ["train", "tune"]:
        train_model(model, train_gen, steps_per_epoch, val_gen,
                    val_steps, args.epochs, learning_rate)

        # Save model in Keras format (h5)
        if args.savemodel:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            raise NotImplementedError('Converting segmentation models is not supported yet.')
        evaluate_model(model, val_gen, steps=val_steps, print_history=True)


if __name__ == "__main__":
    main()
