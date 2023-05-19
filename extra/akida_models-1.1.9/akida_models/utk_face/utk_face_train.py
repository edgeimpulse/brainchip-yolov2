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
Training script for UTKFace model.
"""
import numpy as np

from keras.callbacks import LearningRateScheduler

from cnn2snn import load_quantized_model, convert

from ..training import get_training_parser, compile_model, evaluate_model, print_history_stats
from .preprocessing import load_data


def get_data():
    """ Loads UTKFace data.

    Returns:
        np.array, np.array, np.array, np.array:  train set, train labels, test
            set and test labels
    """
    # Load the dataset
    x_train, y_train, x_test, y_test = load_data()

    return x_train.astype('float32'), y_train, x_test.astype('float32'), y_test


def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        x_train (numpy.ndarray): train data
        y_train (numpy.ndarray): train labels
        x_test (numpy.ndarray): test data
        y_test (numpy.ndarray): test labels
        epochs (int):  the number of epochs
        batch_size (int): the batch size
    """
    # Learning rate: be more aggressive at the beginning, and apply decay
    lr_start = 1e-3
    lr_end = 1e-4
    lr_decay = (lr_end / lr_start)**(1. / epochs)

    lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay**e)
    callbacks = [lr_scheduler]

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    print_history_stats(history)


def main():
    """ Entry point for script and CLI usage.
    """
    parser = get_training_parser(batch_size=128, global_batch_size=False)[0]
    args = parser.parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    compile_model(model, loss='mae', metrics=None)

    # Load data
    x_train, y_train, x_test, y_test = get_data()

    # Train model
    if args.action == "train":
        train_model(model, x_train, y_train, x_test, y_test, args.epochs,
                    args.batch_size)

        # Save model in Keras format (h5)
        if args.savemodel:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            model_ak = convert(model)
            y = model_ak.predict(x_test.astype(np.uint8))
            mae = np.sum(np.abs(y_test.squeeze() - y.squeeze())) / len(y_test)
            print("Validation accuracy: {0:.4f}".format(mae))
        else:
            evaluate_model(model, x_test, y=y_test, print_history=True)


if __name__ == "__main__":
    main()
