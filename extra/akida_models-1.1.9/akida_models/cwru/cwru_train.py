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
Convtiny/CWRU training script.

CWRU is a standard database used in condition monitoring applications. We have
used the drive end data that are sampled at 48 kHz sampling frequency when load
of 1 hp is applied on the shaft. Normal data collected with 1 hp load have also
been used. There are a total of 10 classes. The classes are:

  - C1: Ball defect (0.007 inch, load: 1 hp)
  - C2: Ball defect (0.014 inch, load: 1 hp)
  - C3: Ball defect (0.021 inch, load: 1 hp)
  - C4: Inner race fault (0.007 inch, load: 1 hp)
  - C5: Inner race fault (0.014 inch, load: 1 hp)
  - C6: Inner race fault (0.021 inch, load: 1 hp)
  - C7: Normal (load: 1 hp)
  - C8: Outer race fault (0.007 inch, load: 1 hp, data collected from 6 O'clock
    position)
  - C9: Outer race fault (0.014 inch, load: 1 hp, 6 O'clock)
  - C10: Outer race fault (0.021 inch, load: 1 hp, 6 O'clock)

From each category data are collected in segments of length 1024 and resized to
a 2-D matrix of size (32 by 32). There is no overlap between segments. For each
category 460 such segments are taken. Total size of the data thus becomes (4600,
32, 32). Out of this 1000 segments are randomly chosen as test data and rest are
used for training.
"""

import argparse
import numpy as np

from keras.utils.np_utils import to_categorical

from cnn2snn import load_quantized_model, convert

from ..training import (get_training_parser, compile_model, evaluate_model, print_history_stats,
                        RestoreBest)


def get_data(data_file, input_size):
    """ Loads and prepares data.

    Args:
        data_file (str): file path to the preprocessed npz data
        input_size (int): size to reshape input data

    Returns:
        np.array: four objects containing training data and labels and test data
            and labels
    """
    # Load data file
    raw_data = np.load(data_file)
    data = raw_data['data']
    labels = raw_data['labels']

    # Transform labels strings to categories
    category_dict = {k: v for v, k in enumerate(np.unique(labels).tolist())}
    labels = np.array([category_dict[label] for label in labels], dtype=np.int8)

    # Rescale data to 8-bit, range [0, 255], the model will rescale inputs so
    # that they are zero centered (i.e [-127, 127])
    absmax = np.maximum(data.max(), -data.min())
    data = np.round(data / absmax * 127) + 127

    # Randomly shuffle data
    np.random.seed(42)
    index = np.random.permutation(len(labels))
    data, labels = data[index], labels[index]

    # Split data: there are only 4600 examples in total in CWRU dataset, thus
    # keeping the test set at 1000 examples which is slightly above 21%.
    splitter = 3600
    train_data = data[:splitter]
    train_labels = labels[:splitter]
    test_data = data[splitter:]
    test_labels = labels[splitter:]

    # Reshape data
    train_data = train_data.reshape(len(train_data), input_size, input_size, 1)
    test_data = test_data.reshape(len(test_data), input_size, input_size, 1)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Shuffle data
    index = np.random.permutation(len(train_labels))
    train_data, train_labels = train_data[index], train_labels[index]

    return train_data, train_labels, test_data, test_labels


def train_model(model, train_data, train_labels, batch_size, epochs):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        train_data (np.array): train data
        train_labels (np.array): train labels
        batch_size (int): the batch size
        epochs (int): the number of epochs
    """
    # Define training callbacks
    callbacks = []

    # Model checkpoints (save best model and retrieve it when training is complete)
    restore_model = RestoreBest(model)
    callbacks.append(restore_model)

    history = model.fit(train_data,
                        train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)
    print_history_stats(history)


def main():
    """ Script entry point.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("-d",
                               "--data",
                               type=str,
                               default='./CWRU_48k_load_1_CNN_data.npz',
                               help="File path to the preprocessed npz data.")

    parser = get_training_parser(batch_size=128, global_parser=global_parser)[0]
    args = parser.parse_args()

    # Load the source model
    model = load_quantized_model(args.model)

    # Compile model
    compile_model(model)

    # Load data
    train_data, train_labels, test_data, test_labels = get_data(
        args.data, model.input_shape[1])

    # Train model
    if args.action == "train":
        train_model(model, train_data, train_labels, args.batch_size,
                    args.epochs)

        # Save model in Keras format (h5)
        if args.savemodel:
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        if args.akida:
            model = convert(model)
            accuracy = model.evaluate(test_data.astype(np.uint8),
                                      np.argmax(test_labels, 1))
            print(f'Akida accuracy: {accuracy}')
        else:
            evaluate_model(model,
                           test_data,
                           y=test_labels,
                           batch_size=args.batch_size)


if __name__ == "__main__":
    main()
