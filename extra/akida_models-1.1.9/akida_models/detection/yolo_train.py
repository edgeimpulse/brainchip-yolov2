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
Training script for YOLO models.
"""

import pickle
import argparse
import numpy as np

from keras import Model
from keras.layers import Reshape
from keras.callbacks import EarlyStopping

from cnn2snn import load_quantized_model, convert

from .yolo_loss import YoloLoss
from .map_evaluation import MapEvaluation
from .batch_generator import BatchYoloGenerator
from ..training import get_training_parser, freeze_model_before, compile_model


def get_data(data_path, anchors_path):
    """ Loads data.

    Args:
        data_path (str): path to data pickle file
        anchors_path (str): path to anchors pickle file

    Returns:
        dict, dict, list, list: train and validation data, labels and anchors
    """
    with open(data_path, 'rb') as handle:
        train_data, valid_data, labels = pickle.load(handle)

    with open(anchors_path, 'rb') as handle:
        anchors = pickle.load(handle)

    return train_data, valid_data, labels, anchors


def train(model, train_data, valid_data, anchors, labels, obj_threshold,
          nms_threshold, epochs, batch_size, grid_size):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train
        train_data (dict): training data
        valid_data (dict): validation data
        anchors (list): list of anchors
        labels (list): list of labels
        obj_threshold (float): confidence threshold for a box
        nms_threshold (float): non-maximal supression threshold
        epochs (int): the number of epochs
        batch_size (int): the batch size
        grid_size (tuple): the grid size
    """
    TRAIN_TIMES = 10

    # Build batch generators
    input_shape = model.input.shape[1:]

    train_generator = BatchYoloGenerator(input_shape=input_shape,
                                         data=train_data,
                                         grid_size=grid_size,
                                         labels=labels,
                                         anchors=anchors,
                                         batch_size=batch_size)

    valid_generator = BatchYoloGenerator(input_shape=input_shape,
                                         data=valid_data,
                                         grid_size=grid_size,
                                         labels=labels,
                                         anchors=anchors,
                                         batch_size=batch_size,
                                         jitter=False)

    # Create callbacks
    early_stop_cb = EarlyStopping(monitor='val_loss',
                                  min_delta=0.001,
                                  patience=10,
                                  mode='min',
                                  verbose=1)

    map_evaluator_cb = MapEvaluation(model,
                                     valid_data,
                                     labels,
                                     anchors,
                                     period=4,
                                     obj_threshold=obj_threshold,
                                     nms_threshold=nms_threshold)

    callbacks = [early_stop_cb, map_evaluator_cb]

    # Start the training process
    model.fit(x=train_generator,
              steps_per_epoch=len(train_generator) * TRAIN_TIMES,
              epochs=epochs,
              validation_data=valid_generator,
              callbacks=callbacks,
              workers=12,
              max_queue_size=40)


def evaluate(model, valid_data, anchors, labels, obj_threshold, nms_threshold):
    """ Evaluates model performances.

    Args:
        model (keras.Model or akida.Model): the model to evaluate
        valid_data (dict): evaluation data
        anchors (list): list of anchors
        labels (list): list of labels
        obj_threshold (float): confidence threshold for a box
        nms_threshold (float): non-maximal supression threshold
    """
    # Create the mAP evaluator
    map_evaluator = MapEvaluation(model,
                                  valid_data,
                                  labels,
                                  anchors,
                                  obj_threshold=obj_threshold,
                                  nms_threshold=nms_threshold,
                                  is_keras_model=isinstance(model, Model))

    # Compute mAP scores and display them
    mAP, average_precisions = map_evaluator.evaluate_map()
    for label, average_precision in average_precisions.items():
        print(labels[label], '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(mAP))


def extract_samples(im_size, data, grid_size, anchors, labels, num_samples, out_file):
    """ Extract samples from data and save them to a .npz file.

    Args:
        im_size (int): image size
        data (dict): data container
        grid_size (tuple): the grid size
        anchors (list): list of anchors
        labels (list): list of labels
        num_samples (int): number of samples to extract
        out_file (str): name of output file
    """
    data_generator = BatchYoloGenerator(input_shape=im_size,
                                        data=data,
                                        grid_size=grid_size,
                                        labels=labels,
                                        anchors=anchors,
                                        batch_size=num_samples)
    np.savez(out_file, data_generator[0][0])
    print(f"Samples saved as {out_file}")


def main():
    """ Entry point for script and CLI usage.
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "-d",
        "--data",
        default=None,
        help="The pickle file generated by the preprocessing"
        " step that contains data variables")
    global_parser.add_argument("-ap",
                               "--anchors_path",
                               default=None,
                               help="Path to anchors boxes file.")
    global_parser.add_argument("-obj",
                               "--obj_thresh",
                               type=float,
                               default=0.5,
                               help="Confidence threshold for a box")
    global_parser.add_argument("-nms",
                               "--nms_thresh",
                               type=float,
                               default=0.5,
                               help="Non-Maximal Suppression threshold.")

    parsers = get_training_parser(batch_size=128,
                                  freeze_before=True,
                                  tune=True,
                                  global_parser=global_parser)
    subparser = parsers[-1]
    extract_parser = subparser.add_parser("extract",
                                          help="Extract samples",
                                          parents=[global_parser])
    extract_parser.add_argument("-o", "--out_file", type=str,
                                default='widerface_samples.npz', help="Name of the output file.")

    args = parsers[0].parse_args()

    # Load data
    train_data, valid_data, labels, anchors = get_data(args.data,
                                                       args.anchors_path)

    # Load the source model
    base_model = load_quantized_model(args.model)

    # Create a final reshape layer for loss computation
    grid_size = base_model.output_shape[1:3]
    output = Reshape(
        (grid_size[1], grid_size[0], len(anchors), 4 + 1 + len(labels)),
        name="YOLO_output")(base_model.output)

    # Build the full model
    model = Model(base_model.input, output)

    # Freeze the model
    if "freeze_before" in args:
        freeze_model_before(model, args.freeze_before)

    # Compile model
    learning_rate = 1e-4
    if args.action == "tune":
        learning_rate = 1e-5

    compile_model(model,
                  learning_rate=learning_rate,
                  loss=YoloLoss(anchors, grid_size, args.batch_size),
                  metrics=None)

    # Train model
    if args.action in ["train", "tune"]:
        train(model, train_data, valid_data, anchors, labels, args.obj_thresh,
              args.nms_thresh, args.epochs, args.batch_size, grid_size)

        if args.savemodel:
            # Remove the last reshape layer introduced for training
            model = Model(model.input, model.layers[-2].output)
            # Save the model
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == 'eval':
        # Evaluate model accuracy
        if args.akida:
            # Drop the last reshape layer that is not Akida compatible
            if model.layers[-1].name == 'YOLO_output':
                model = Model(model.input, model.layers[-2].output)
            model = convert(model)
        evaluate(model, valid_data, anchors, labels, args.obj_thresh,
                 args.nms_thresh)

    elif args.action == 'extract':
        input_shape = model.input.shape[1:]
        extract_samples(input_shape, train_data, grid_size, anchors,
                        labels, args.batch_size, args.out_file)


if __name__ == "__main__":
    main()
