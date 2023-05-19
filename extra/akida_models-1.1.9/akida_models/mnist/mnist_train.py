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
Training script for MNIST models.
"""

from keras.datasets import mnist
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.losses import SparseCategoricalCrossentropy

from cnn2snn import load_quantized_model, convert

import numpy as np

from ..distiller import Distiller, KLDistillationLoss
from ..training import get_training_parser, compile_model, evaluate_model, print_history_stats


def get_data():
    """ Loads MNIST data.

    Returns:
        tuple: train data, train labels, test data and test labels
    """
    # The data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32")
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32")
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return x_train, y_train, x_test, y_test


def train_model(model, teacher_model, x_train, y_train, batch_size, epochs):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        teacher_model (keras.Model): the teacher model used for the Knowledge
            distillation training
        x_train (numpy.ndarray): train data
        y_train (numpy.ndarray): train labels
        batch_size (int): the batch size
        epochs (int): the number of epochs
    """
    # Create a distiller
    distiller = Distiller(student=model, teacher=teacher_model, alpha=0.1)

    # Compile the distiller
    distiller.compile(
        optimizer=Adam(learning_rate=5e-5),
        metrics=[SparseCategoricalAccuracy()],
        student_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=KLDistillationLoss(temperature=10))

    # Train the model
    history = distiller.fit(x_train, y_train, batch_size, epochs)
    print_history_stats(history)


def train_model_classical(model, x_train, y_train, batch_size, epochs):
    """ Trains the model without the distillation.

    Args:
        model (keras.Model): the model to train
        x_train (numpy.ndarray): train data
        y_train (numpy.ndarray): train labels
        batch_size (int): the batch size
        epochs (int): the number of epochs
    """
    history = model.fit(x_train, y_train, batch_size, epochs)
    print_history_stats(history)


def main():
    """ Entry point for script and CLI usage.
    """
    parsers = get_training_parser(batch_size=32, global_batch_size=False)

    train_parser = parsers[1]
    train_parser.add_argument("-t",
                              "--teacher",
                              type=str,
                              default=None,
                              help="Teacher model to use for the training")

    args = parsers[0].parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    compile_model(model,
                  learning_rate=5e-5,
                  loss=SparseCategoricalCrossentropy(from_logits=True))

    # Load data
    x_train, y_train, x_test, y_test = get_data()

    # Train model
    if args.action == "train":
        if args.teacher is not None:
            # Load the teacher model
            teacher_model = load_quantized_model(args.teacher)

            train_model(model, teacher_model, x_train, y_train, args.batch_size,
                        args.epochs)
        else:
            train_model_classical(model, x_train, y_train, args.batch_size,
                                  args.epochs)

        # Save Model in Keras format (h5)
        if args.savemodel:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            model_ak = convert(model)
            accuracy = model_ak.evaluate(x_test.astype('uint8'), y_test)
            print(f"Accuracy: {accuracy}")
        else:
            evaluate_model(model, x_test, y=y_test)


if __name__ == "__main__":
    main()
