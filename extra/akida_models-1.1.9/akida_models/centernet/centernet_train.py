#!/usr/bin/env python
# ******************************************************************************
# Copyright 2023 Brainchip Holdings Ltd.
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
Training script for CenterNet models
"""
import pickle
import argparse

from keras import Model

from ..training import get_training_parser, freeze_model_before, compile_model
from ..detection.map_evaluation import MapEvaluation
from ..param_scheduler import get_cosine_lr_scheduler
from ..utils import get_tensorboard_callback, load_model

from .centernet_batch_generator import BatchCenternetGenerator
from .centernet_loss import CenternetLoss
from .centernet_processing import decode_output


def get_data(data_path):
    """Loads data.

    Args:
        data_path (str): path to data pickle file

    Returns:
        dict, dict, list: train and validation data, labels
    """
    with open(data_path, 'rb') as handle:
        train_data, valid_data, labels = pickle.load(handle)
    return train_data, valid_data, labels


def _decode_centernet_output_fn(output, anchors, nb_classes, obj_threshold=0.5, nms_threshold=0.5):
    # Ignore attributes from original MAPEvaluator
    return decode_output(output, nb_classes, obj_threshold=obj_threshold)


def train(model, train_data, valid_data, labels, lr_max, obj_threshold,
          epochs, batch_size, grid_size, tune=False, out_dir=None):
    """ Trains the model.

    Args:
        model (keras.Model): the model to train.
        train_data (dict): training data.
        valid_data (dict): validation data.
        labels (list): classes name.
        lr_max (float): max learning rate to take in learning rate scheduler.
        obj_threshold (float): confidence threshold for inference procedure.
        epochs (int): the number of epochs.
        batch_size (int): the batch size.
        grid_size (tuple): the spatial hw output size.
        tune (bool, optional): wheter the model will be tuned or not, modifying ``lr_max``.
            Defaults to False.
        out_dir (str, optional): folder name to save logs. Defaults to None
    """
    TRAIN_TIMES = 10

    # Build batch generators
    input_shape = model.input.shape[1:]

    # create the data generators
    train_generator = BatchCenternetGenerator(input_shape=input_shape,
                                              data=train_data,
                                              grid_size=grid_size,
                                              labels=labels,
                                              batch_size=batch_size)

    valid_generator = BatchCenternetGenerator(input_shape=input_shape,
                                              data=valid_data,
                                              grid_size=grid_size,
                                              labels=labels,
                                              batch_size=batch_size,
                                              jitter=False)

    # Create callbacks
    steps_per_epoch = len(train_generator) * TRAIN_TIMES
    map_evaluator_cb = MapEvaluation(model,
                                     valid_data,
                                     labels,
                                     anchors=None,
                                     period=4,
                                     obj_threshold=obj_threshold,
                                     decode_output_fn=_decode_centernet_output_fn)
    lrs_callback = get_cosine_lr_scheduler(lr_max,
                                           steps_per_epoch * epochs,
                                           pct_start=0.001 if tune else 0.3)
    tensorboard = get_tensorboard_callback(out_dir, prefix="centernet")

    callbacks = [map_evaluator_cb, lrs_callback, tensorboard]

    # Start the training process
    model.fit(x=train_generator,
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              validation_data=valid_generator,
              callbacks=callbacks,
              workers=12,
              max_queue_size=40)


def evaluate(model, valid_data, labels, obj_threshold):
    """ Evaluates model performances.

    Args:
        model (keras.Model or akida.Model): the model to evaluate.
        valid_data (dict): evaluation data.
        labels (list): classes name.
        obj_threshold (float): confidence threshold for inference procedure.
    """
    # Create the mAP evaluator
    map_evaluator = MapEvaluation(model,
                                  valid_data,
                                  labels,
                                  anchors=None,
                                  obj_threshold=obj_threshold,
                                  is_keras_model=isinstance(model, Model),
                                  decode_output_fn=_decode_centernet_output_fn)

    # Compute mAP scores and display them
    mAP, average_precisions = map_evaluator.evaluate_map()
    for label, average_precision in average_precisions.items():
        print(labels[label], '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(mAP))


def main():
    """ Entry point for script and CLI usage. """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("-d", "--data", default=None,
                               help="The pickle file generated by the preprocessing step")
    global_parser.add_argument("-obj", "--obj_thresh", type=float, default=0.1,
                               help="Confidence threshold for a box. Defaults to %(default)s")
    global_parser.add_argument("-o", "--out_dir", type=str, default=None,
                               help="The output directory (logs). Defaults to %(default)s")
    parsers = get_training_parser(batch_size=128, freeze_before=True, tune=True,
                                  global_parser=global_parser)
    args = parsers[0].parse_args()

    # Load data
    train_data, valid_data, labels = get_data(args.data)

    # Load the source model
    model = load_model(args.model)

    grid_size = model.output.shape[1:3]

    nlabels, nmodel = len(labels), model.output.shape[-1]
    if nlabels + 4 != nmodel:
        raise ValueError(f"Model's output ({nmodel}) does not match with "
                         f"number of labels ({nlabels}) + 4. "
                         f"Check the data file {args.data} or input model {args.model}.")
    # Freeze the model
    if "freeze_before" in args:
        freeze_model_before(model, args.freeze_before)

    # Compile model
    learning_rate = 1e-4 if args.action == "tune" else 1e-2
    compile_model(model, learning_rate=learning_rate, loss=CenternetLoss(), metrics=None)

    if args.action in ["train", "tune"]:
        train(model, train_data, valid_data, labels, learning_rate, args.obj_thresh,
              args.epochs, args.batch_size, grid_size, out_dir=args.out_dir,
              tune=args.action == "tune")

        if args.savemodel:
            # Save the model
            model.save(args.savemodel, include_optimizer=False)
            print(f"Trained model saved as {args.savemodel}")

    elif args.action == "eval":
        # Evaluate model accuracy
        # TODO: enable this feature when cnn2snn support SeparableConv2DTranspose
        # and full conversion into akida
        if args.akida:
            raise NotImplementedError("Not implemented yet!")
        evaluate(model, valid_data, labels, args.obj_thresh)


if __name__ == "__main__":
    main()
