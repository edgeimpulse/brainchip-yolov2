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
Common utility methods used by the training scripts.
"""

import os.path
import tempfile
import argparse
import numpy as np

from keras.preprocessing.image import DirectoryIterator
from keras.optimizers import Adam
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar
from quantizeml.layers import calibration


def compile_model(model,
                  learning_rate=1e-3,
                  loss='categorical_crossentropy',
                  metrics=None):
    """ Compiles the model using Adam optimizer.

    Args:
        model (keras.Model): the model to compile
        learning_rate (float, optional): the learning rate. Defaults to 1e-3.
        loss (str or function, optional): the loss function. Defaults to
            'categorical_crossentropy'.
        metrics (list, optional): list of metrics to be evaluated during
            training and testing. Defaults to None.
    """
    if metrics is None:
        metrics = ['accuracy']

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=metrics)


def freeze_model_before(model, freeze_before):
    """ Freezes the model before the given layer name.

    Args:
        model (keras.Model): the model to freeze
        freeze_before (str): name of the layer from which the model will not be
            frozen

    Raises:
        ValueError: if the provided layer name was not found in the model
    """
    if freeze_before is not None:
        trainable = False
        for layer in model.layers:
            if layer.name == freeze_before:
                trainable = True
            layer.trainable = trainable
        if not trainable:
            raise ValueError(f"No such layer {freeze_before} in model.")


def calibrate_model(model, x, num_samples, batch_size=None, epochs=1):
    """Calibrates model

    Args:
        model (keras.Model): the model to calibrate
        x (tf.Dataset, np.array or generator): train input data
        num_samples (int): total number of samples before declaring the calibration round finished.
        batch_size (int, optional): the batch size. Defaults to None.
        epochs (int, optional): the number of epochs. Defaults to 1.
    """
    if batch_size is None:
        steps = num_samples
    else:
        assert batch_size > 0, "The batch size should be strictly positive."
        steps = np.ceil(num_samples / batch_size)

    with calibration(True):
        progbar = Progbar(epochs)
        for epoch in range(epochs):
            progbar.update(epoch)
            model.predict(x=x, steps=steps, batch_size=batch_size, verbose=0)
        progbar.update(epochs, finalize=epochs)


def evaluate_model(model,
                   x,
                   y=None,
                   batch_size=None,
                   steps=None,
                   print_history=False):
    """ Evaluates model performances.

    Args:
        model (keras.Model): the model to evaluate
        x (tf.Dataset, np.array or generator): evaluation input data
        y (tf.Dataset, np.array or generator, optional): evaluation target
            data. Defaults to None.
        batch_size (int, optional): the batch size. Defaults to None.
        steps (int, optional): total number of steps before declaring the
            evaluation round finished. Defaults to None.
        print_history (bool, optional): either to print all history or just
            accuracy. Defaults to False.
    """
    history = model.evaluate(x=x,
                             y=y,
                             batch_size=batch_size,
                             steps=steps,
                             verbose=0)

    if print_history:
        print('Validation history: ', history)
    else:
        print('Validation accuracy: ', history[1])


def evaluate_akida_model(model, x, activation):
    """ Evaluates Akida model and return predictions and labels to compute
    accuracy.

    Args:
        model (akida.Model): the model to evaluate
        x (tf.Dataset, np.array or generator): evaluation input data
        activation (str): activation function to apply to potentials

    Returns:
        np.array, np.array: predictions and labels
    """

    # Initialize to None to allow different shapes depending on the caller
    labels = None
    pots = None

    # Datasets created as ImageDataGenerator will be looped over indefinitely,
    # they thus need a special treatment
    steps = None
    if isinstance(x, DirectoryIterator):
        steps = np.math.ceil(x.samples / x.batch_size)

    for batch, label_batch in x:
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()

        pots_batch = model.predict(batch.astype('uint8'))

        if labels is None:
            labels = label_batch
            pots = pots_batch.squeeze(axis=(1, 2))
        else:
            labels = np.concatenate((labels, label_batch))
            pots = np.concatenate((pots, pots_batch.squeeze(axis=(1, 2))))

        # End the for loop if the number of batches has reached the number of
        # steps
        if steps is not None and x.total_batches_seen == steps:
            break

    return Activation(activation)(pots), labels


def get_training_parser(batch_size,
                        calibrate=False,
                        tune=False,
                        freeze_before=False,
                        global_batch_size=True,
                        global_parser=None):
    """ Instantiates a base arguments parser for training scripts.

    The returned parser comes with train, tune and eval actions as subparsers
    (with default parameters) for which one can add arguments if required.

    Args:
        batch_size (int): batch size default value
        calibrate (bool, optional): either to add a calibrate action or not. Defaults to
            False.
        tune (bool, optional): either to add a tune action or not. Defaults to
            False.
        freeze_before (bool, optional): either to add a "--freeze_before"
            parameter in train and tune actions or not. Defaults to False.
        global_batch_size (bool, optional): either to add batch size as a
            global parser parameter or for train/tune actions only. Defaults to
            True.
        global_parser (argparse.ArgumentParser, optional): global parser with
            custom parameters. Defaults to None.

    Returns:
        argparse.ArgumentParser, argparse.ArgumentParser,
        argparse.ArgumentParser, argparse.ArgumentParser,
        argparse.ArgumentParser,
        argparse._SubParsersAction: main parser and train, calibrate, tune and eval actions
            subparsers and the subparser object.
    """

    # Define a strictly positive int type for parameters
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid value, must be >0." % value)
        return ivalue

    parser = argparse.ArgumentParser()

    # Base arguments
    if global_parser is None:
        global_parser = argparse.ArgumentParser(add_help=False)

    global_parser.add_argument("-m",
                               "--model",
                               type=str,
                               required=True,
                               help="Model to load")
    if global_batch_size:
        global_parser.add_argument("-b",
                                   "--batch_size",
                                   type=positive_int,
                                   default=batch_size,
                                   help="The batch size")

    # Create an action subparser
    subparsers = parser.add_subparsers(help="Desired action to perform",
                                       dest="action")

    # Create parent subparser for arguments shared between train and tune
    # actions
    parent_parser = argparse.ArgumentParser(add_help=False,
                                            parents=[global_parser])
    parent_parser.add_argument("-s",
                               "--savemodel",
                               type=str,
                               default=None,
                               help="Save model with the specified name")
    parent_parser.add_argument("-e",
                               "--epochs",
                               type=positive_int,
                               required=True,
                               help="The number of training epochs")
    if not global_batch_size:
        parent_parser.add_argument("-b",
                                   "--batch_size",
                                   type=positive_int,
                                   default=batch_size,
                                   help="The batch size")
    if freeze_before:
        parent_parser.add_argument(
            "-fb",
            "--freeze_before",
            type=str,
            default=None,
            help="Freeze the layers of the model before this \
                                layer")

    # Subparsers based on parent
    train_parser = subparsers.add_parser("train",
                                         parents=[parent_parser],
                                         help="Train a Keras model")
    tune_parser = None
    if tune:
        tune_parser = subparsers.add_parser("tune",
                                            parents=[parent_parser],
                                            help="Tune a Keras model")

    # Other subparsers
    calibrate_parser = None
    if calibrate:
        calibrate_parser = subparsers.add_parser("calibrate",
                                                 parents=[parent_parser],
                                                 help="Calibrate a Keras model")
        calibrate_parser.add_argument("-ns",
                                      "--num_samples",
                                      type=positive_int,
                                      default=1024,
                                      help="Number of samples to use for calibration")

    eval_parser = subparsers.add_parser("eval",
                                        help="Evaluate a model",
                                        parents=[global_parser])

    eval_parser.add_argument("-ak",
                             "--akida",
                             action='store_true',
                             help="Converts to an Akida model and evaluate")

    return parser, train_parser, tune_parser, eval_parser, calibrate_parser, subparsers


def print_history_stats(history):
    """Prints a basic statistics of the training/eval history.

    Args:
        history (keras.callbacks.History): history of the training/eval process
    """
    print("\nHistory statistics:\n" + "-" * 80 + "\n")
    max_len_metric = max(len(metric) for metric in history.history.keys())
    clean_history = {key.ljust(max_len_metric): values for key, values in history.history.items()}
    stats = [np.min, np.max, np.mean, np.std]
    print("Parameter:".ljust(max_len_metric) + "\t" + "\t".join(f.__name__.ljust(6) for f in stats))
    for key, values in clean_history.items():
        values = "\t".join(f"{f(values):.4f}" for f in stats)
        print(f"{key}\t{values}")


class RestoreBest(ModelCheckpoint):
    """Keep best weights on training

    This tool uses a ModelCheckPoint callback to save the best weights during the training and
    restore them in the model when training is complete. The best model is saved to a
    temporary directory. So if the training does not go to the end, this model can be used for
    start a new one.

    Args:
        model (keras.Model): the model to load the best weights
        monitor (str, optional): the metric name to monitor. Defaults to 'val_accuracy'.
        mode (str, optional): one of {'auto', 'min', 'max'}. The decision to overwrite the current
            save model is made based on either the maximization or the minimization of the monitored
            quantity. Defaults to 'max'.
    """
    def __init__(self, model, monitor='val_accuracy', mode='max'):
        self.model = model
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self.temp_dir.name, "best_model.h5")
        print(f"Best model saved as {self.filepath}.")
        super().__init__(filepath=self.filepath,
                         monitor=monitor,
                         save_best_only=True,
                         mode=mode)

    def on_train_end(self, logs=None):
        if os.path.exists(self.filepath):
            self.model.load_weights(self.filepath)
        # Clean temp directory
        self.temp_dir.cleanup()
