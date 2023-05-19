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
ModelNet40 model training script.
"""

import argparse
import keras
import numpy as np

from keras.callbacks import LearningRateScheduler

from cnn2snn import load_quantized_model, convert

from .preprocessing import get_modelnet, get_modelnet_from_file

from ..training import get_training_parser, compile_model, print_history_stats


def get_data(batch_size, selected_points, knn_points):
    """ Loads data.

    Args:
        batch_size (int): the batch size
        selected_points (int): num points to process per sample.
        knn_points (int): number of points to include in each localised group.
            Must be a power of 2, and ideally an integer square (so 64, or 16
            for a deliberately small network, or 256 for large).

    Returns:
        tf.Dataset, tf.Dataset: train and test point with data augmentation
    """
    # This is the number of points sampled from each input file,
    # when parsing the data, and done only once (and saved as a .npz file)
    # for efficiency.
    train_points, train_labels, test_points, test_labels, _ = \
        get_modelnet_from_file(num_points=1024)

    train_dataset, test_dataset = get_modelnet(train_points,
                                               train_labels,
                                               test_points,
                                               test_labels,
                                               batch_size=batch_size,
                                               selected_points=selected_points,
                                               knn_points=knn_points)

    return train_dataset, test_dataset


def train_model(model, train_dataset, test_dataset, epochs):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train.
        train_dataset (tf.dataset): train data.
        test_dataset (tf.dataset): test data.
        epochs (int): the number of epochs.
    """
    # Learning rate scheduler
    callbacks = []
    max_lr = 0.001
    min_lr = max_lr * 0.01
    lr_decay = (min_lr / max_lr)**(1. / epochs)
    lr_scheduler = LearningRateScheduler(lambda e: max_lr * lr_decay**e)
    callbacks.append(lr_scheduler)

    history = model.fit(train_dataset,
                        validation_data=test_dataset,
                        epochs=epochs,
                        callbacks=callbacks)
    print_history_stats(history)


def evaluate_model(model, test_dataset):
    """ Evaluates model performances.

    Args:
        model (keras.Model or akida.Model): the model to evaluate.
        test_dataset (tf.dataset): evaluation dataset.
    """
    labels = np.concatenate([labels for x, labels in test_dataset], axis=0)

    if isinstance(model, keras.Model):
        preds = np.squeeze(np.argmax(model.predict(test_dataset), axis=1))
    else:
        pots_akida = np.empty(shape=[0, model.output_shape[-1]],
                              dtype=np.float32)

        for batch, _ in test_dataset:
            # Rescale inputs to Akida
            pots = (batch.numpy() * 127 + 127).astype('uint8')
            pots_batch = model.predict(pots)
            pots_akida = np.concatenate((pots_akida, pots_batch.squeeze()))

        preds = np.squeeze(np.argmax(pots_akida, 1))

    acc = np.sum(preds == labels) / labels.shape[0]
    print(f"Accuracy: {acc}")


def main():
    """ Entry point for script and CLI usage.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "-p",
        "--selected_points",
        type=int,
        default=128,
        help="The number of points to process per sample. \
                        (default 128)")
    global_parser.add_argument("-n",
                               "--knn_points",
                               type=int,
                               default=64,
                               help="The number of points to include in each \
                        localised group. (default 64)")

    parser = get_training_parser(batch_size=16, global_parser=global_parser)[0]
    args = parser.parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    compile_model(model, loss="sparse_categorical_crossentropy")

    # Load data
    train_dataset, test_dataset = get_data(args.batch_size,
                                           args.selected_points,
                                           args.knn_points)

    # Train model
    if args.action == "train":
        train_model(model, train_dataset, test_dataset, args.epochs)

        # Save model in Keras format (h5)
        if args.savemodel is not None:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            model = convert(model, input_scaling=(127, 127))
        evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
