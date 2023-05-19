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
Training script for DVS models.
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from keras.utils.np_utils import to_categorical

from cnn2snn import load_quantized_model, convert

from ..cyclic_lr import CyclicLR
from ..training import (get_training_parser, compile_model, evaluate_model, evaluate_akida_model,
                        print_history_stats, RestoreBest)
from ..utils import fetch_file


def get_data(dataset, batch_size):
    """ Loads data.

    Dataset parameters for DVSDataGenerator are set from DVS pickles.

    Args:
        dataset (str): name of the dataset to load, either 'dvs_gesture' or
            'samsung_handy'.
        batch_size (int): the batch size

    Returns:
        tf.dataset, tf.dataset, int: train dataset, test dataset and number of
        samples.
    """
    # Load pre-processed dataset
    train_file = fetch_file(os.path.join('http://data.brainchip.com/dataset-mirror',
                                         dataset, dataset + '_preprocessed_train.npz'),
                            fname=dataset + '_preprocessed_train.npz',
                            cache_subdir=os.path.join('datasets', dataset))

    test_file = fetch_file(os.path.join('http://data.brainchip.com/dataset-mirror',
                                        dataset, dataset + '_preprocessed_test.npz'),
                           fname=dataset + '_preprocessed_test.npz',
                           cache_subdir=os.path.join('datasets', dataset))
    train_data = np.load(train_file)
    x_train = train_data['x_train']
    y_train = to_categorical(train_data['y_train'])

    test_data = np.load(test_file)
    x_test = test_data['x_test']
    y_test = to_categorical(test_data['y_test'])

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05, interpolation='nearest'),
    ])

    train_dataset = train_dataset.shuffle(
        buffer_size=1024).batch(batch_size).map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE).prefetch(
                buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset, x_train.shape[0]


def train_model(model, train_ds, test_ds, batch_size, max_learning_rate, epochs,
                num_samples):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        train_ds (tf.dataset): train dataset
        test_ds (tf.dataset): test dataset
        batch_size (int): the batch size
        max_learning_rate (float): learning rate maximum value
        epochs (int): the number of epochs
        num_samples (int): number of samples
    """
    n_iterations = np.round(num_samples / batch_size)

    # Define training callbacks
    callbacks = []

    lr_scheduler = CyclicLR(base_lr=max_learning_rate * 0.01,
                            max_lr=max_learning_rate,
                            step_size=4 * n_iterations,
                            mode='triangular2')
    callbacks.append(lr_scheduler)

    # Model checkpoints (save best model and retrieve it when training is complete)
    restore_model = RestoreBest(model)
    callbacks.append(restore_model)

    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=test_ds)
    print_history_stats(history)


def main():
    """ Entry point for script and CLI usage.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("-d",
                               "--dataset",
                               type=str,
                               default='dvs_gesture',
                               choices=['dvs_gesture', 'samsung_handy'],
                               help="Dataset name (defaut=dvs_gesture)")

    parsers = get_training_parser(batch_size=32,
                                  tune=True,
                                  global_parser=global_parser)
    args = parsers[0].parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    learning_rate = 0.001
    if args.action == 'tune':
        learning_rate = 0.0001

    compile_model(model, learning_rate=learning_rate)

    # Load data
    train_ds, test_ds, num_samples = get_data(args.dataset, args.batch_size)

    # Train model
    if args.action in ["train", "tune"]:
        train_model(model, train_ds, test_ds, args.batch_size, learning_rate,
                    args.epochs, num_samples)

        # Save model in Keras format (h5)
        if args.savemodel:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            model_ak = convert(model, input_is_image=False)
            preds, labels = evaluate_akida_model(model_ak, test_ds, 'softmax')
            accuracy = (np.squeeze(np.argmax(preds, 1)) == np.argmax(labels,
                                                                     1)).mean()
            print(f"Akida accuracy: {accuracy}")
        else:
            evaluate_model(model, test_ds)


if __name__ == "__main__":
    main()
