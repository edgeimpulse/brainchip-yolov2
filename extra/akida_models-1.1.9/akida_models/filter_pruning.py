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
"""Filter pruning tools to automatically prune a model"""

import numpy as np
from keras import Sequential, Input
from keras.layers import (InputLayer, Conv2D, SeparableConv2D, Dense,
                          BatchNormalization, MaxPool2D, GlobalAvgPool2D,
                          Dropout, ReLU, Softmax, Activation, Flatten, Reshape,
                          Rescaling)

import cnn2snn
from cnn2snn import QuantizedConv2D, QuantizedSeparableConv2D, QuantizedDense


def delete_filters(model, layer_to_prune, filters_to_prune):
    """Deletes filters in the given layer and updates weights in it and its
    subsequent layers.

    A pruned model is returned. Only linear models are supported.

    Args:
        model (keras.Model): the model to prune.
        layer_to_prune (str): the name of the neural layer where filters will be
            deleted.
        filters_to_prune (list): indices of filters to delete in the given
            layer.

    Returns:
        keras.Sequential: the pruned model
    """

    pruned_model = _instantiate_pruned_model(model, layer_to_prune,
                                             len(filters_to_prune))
    _assign_pruned_weights(model, pruned_model, layer_to_prune,
                           filters_to_prune)

    return pruned_model


def _instantiate_pruned_model(model, layer_to_prune, num_filters_to_prune):
    """Returns the pruned model where the layer to prune has less filters than
    the original layer. Note that the weights are not transferred yet
    from the original model to the pruned one.

    Args:
        model (keras.Model): the model to prune.
        layer_to_prune (str): the name of the layer where filters will be
            deleted.
        num_filters_to_prune (int): number of filters to delete in the given
            layer.

    Returns:
        keras.Sequential: the pruned model (without weights from the original
            model)
    """

    pruned_model = Sequential()
    pruned_model.add(Input(model.input_shape[1:]))

    # Create the pruned model by browsing the layers of the original model and
    # by deleting filters/channels in the concerned layers:
    # - for each layer of the original model, create a new layer with the new
    #   configuration
    # - add the new layer to the pruned Sequential model
    for layer in model.layers:

        if isinstance(layer, InputLayer):
            continue

        if isinstance(layer, (Conv2D, SeparableConv2D)):
            new_config = layer.get_config()
            if layer.name == layer_to_prune:
                new_config[
                    'filters'] = new_config['filters'] - num_filters_to_prune
                if new_config['filters'] == 0:
                    raise RuntimeError("Impossible to prune all filters of "
                                       f"layer {layer_to_prune}.")
            new_layer = layer.__class__.from_config(new_config)

        elif isinstance(layer, Dense):
            new_config = layer.get_config()
            if layer.name == layer_to_prune:
                new_config['units'] = new_config['units'] - num_filters_to_prune
                if new_config['units'] == 0:
                    raise RuntimeError("Impossible to prune all units of "
                                       f"layer {layer_to_prune}.")
            new_layer = layer.__class__.from_config(new_config)

        elif isinstance(layer, Reshape):
            new_config = layer.get_config()
            # If the layer preceding a Reshape layer was pruned, the new Reshape
            # layer must handle the new target shape. Note that, in cnn2snn,
            # only Reshape layer (N,)->(1,1,N) is supported.
            if layer.target_shape[-1] != pruned_model.output_shape[-1]:
                new_target_shape = layer.target_shape[:-1] + (
                    pruned_model.output_shape[-1],)
                new_config['target_shape'] = new_target_shape
            new_layer = layer.__class__.from_config(new_config)

        elif isinstance(layer, (BatchNormalization, MaxPool2D, GlobalAvgPool2D,
                                Dropout, Flatten, ReLU, Softmax, Activation,
                                Rescaling, cnn2snn.QuantizedActivation)):
            new_layer = layer.__class__.from_config(layer.get_config())

        else:
            raise RuntimeError(
                f"Layer {layer.name} of type {layer.__class__.__name__} "
                "is not supported for pruning")

        pruned_model.add(new_layer)

    return pruned_model


def _assign_pruned_weights(model, pruned_model, layer_to_prune,
                           filters_to_prune):
    """Assigns new weights from the original model to the pruned one.

    For each layer in the model, two steps are performed:
        - update the outbound mask. The outbound mask (of length the number of
          output channels) represents the filters/channels to keep.
        - set the new pruned weights, according to the inbound/outbound masks.

    Args:
        model (keras.Model): the model to prune.
        pruned_model (keras.Sequential): the pruned model where the weights
            must be set.
        layer_to_prune (str): the name of the layer where filters were deleted.
        filters_to_prune (list): indices of filters to delete in the given
            layer.
    """

    # Mask representing the filters to keep (False = filters to prune)
    inbound_mask = [True] * model.input_shape[-1]

    for layer in model.layers:

        # Step 1: Update the outbound mask.
        # Only layers that modifies the number of channels are concerned (i.e.
        # neural layers and Flatten layers). The mask for the layer to prune has
        # "False" values for indices to prune. The mask for other neural layers
        # is "True" for all output channels.
        if layer.name == layer_to_prune:
            outbound_mask = [
                i not in filters_to_prune for i in range(layer.output_shape[-1])
            ]

        elif isinstance(layer, (Conv2D, SeparableConv2D, Dense)):
            # All other (i.e. not pruned) neural layers reset the outbound_mask
            # (all filters are kept)
            outbound_mask = [True] * layer.output_shape[-1]

        elif isinstance(layer, Flatten):
            # A Flatten layer modifies the number of channels: C_out = H*W*C_in.
            # The outbound mask is updated, depending on the input size:
            # input_size = H*W = C_out/C_in
            input_size = int(layer.output_shape[-1] / layer.input_shape[-1])
            outbound_mask = inbound_mask * input_size

        elif isinstance(layer,
                        (InputLayer, BatchNormalization, MaxPool2D,
                         GlobalAvgPool2D, Dropout, ReLU, Softmax, Activation,
                         Reshape, Rescaling, cnn2snn.QuantizedActivation)):
            # All these layers have no weights and do not change the number of
            # channels. Reshape could modify the number of channels. However we
            # only support (N,) -> (1,1,N), thus there is no change in the
            # number of channels.
            outbound_mask = inbound_mask

        else:
            raise RuntimeError(
                f"Layer {layer.name} of type {layer.__class__.__name__} "
                "is not supported for pruning")

        # Step 2: set new pruned weights.
        # The concerned layers are all layers between the neural layer to prune
        # and the next neural layer. Layers with weights are
        # (Quantized)Conv2D/SeparableConv2D/Dense and BatchNormalization layers.
        if isinstance(layer, Conv2D):
            new_weights = layer.get_weights()
            new_weights[0] = new_weights[0][:, :, inbound_mask, :]
            new_weights[0] = new_weights[0][..., outbound_mask]
            if layer.use_bias:
                new_weights[1] = new_weights[1][outbound_mask]
            pruned_model.get_layer(layer.name).set_weights(new_weights)

        elif isinstance(layer, SeparableConv2D):
            new_weights = layer.get_weights()
            new_weights[0] = new_weights[0][:, :, inbound_mask, :]
            new_weights[1] = new_weights[1][:, :, inbound_mask, :]
            new_weights[1] = new_weights[1][..., outbound_mask]
            if layer.use_bias:
                new_weights[2] = new_weights[2][outbound_mask]
            pruned_model.get_layer(layer.name).set_weights(new_weights)

        elif isinstance(layer, Dense):
            new_weights = layer.get_weights()
            new_weights[0] = new_weights[0][inbound_mask, :]
            new_weights[0] = new_weights[0][:, outbound_mask]
            if layer.use_bias:
                new_weights[1] = new_weights[1][outbound_mask]
            pruned_model.get_layer(layer.name).set_weights(new_weights)

        elif isinstance(layer, BatchNormalization):
            outbound_mask = inbound_mask
            new_weights = layer.get_weights()
            for i in range(4):
                new_weights[i] = new_weights[i][inbound_mask]
            pruned_model.get_layer(layer.name).set_weights(new_weights)

        # Update inbound mask for next layer
        inbound_mask = outbound_mask


def neural_layers(model):
    """ Returns the list of neural layers names, except the last one, in reverse
    order. The idea is to prune the last layers first.

    Args:
        model (keras.Model): the model to prune.

    Returns:
        list: the names of the prunable layers.
    """
    layers = [
        layer.name
        for layer in model.layers
        if type(layer) in (Conv2D, SeparableConv2D, Dense, QuantizedConv2D,
                           QuantizedSeparableConv2D, QuantizedDense)
    ]

    # exclude the last layer and reverse the order
    return layers[::-1][1:]


def _get_weights_for_pruning(layer):
    """Returns the kernel of the layer where filters will be pruned.
    For SeparableConv2D, it returns the pointwise kernel.

    Args:
        layer (keras.Layer): the layer to prune.

    Returns:
        `:obj:np.ndarray`: the weights used for pruning.
    """
    if isinstance(layer, (Conv2D, QuantizedConv2D, Dense, QuantizedDense)):
        weights_index = 0
    elif isinstance(layer, (SeparableConv2D, QuantizedSeparableConv2D)):
        weights_index = 1
    else:
        raise RuntimeError(
            f"Layer {layer.name} must be of type (Quantized)Conv2D, "
            "(Quantized)SeparableConv2D, "
            f"(Quantized)Dense. Receives type {type(layer)}.")
    return layer.get_weights()[weights_index]


def _filter_magnitudes(layer):
    """Returns the absolute mean of filters for the given layer. This is called
    the magnitude of a filter.

    Args:
        layer (keras.Layer): the layer to prune.

    Returns:
        `:obj:np.ndarray`: the magnitudes of the layer's filters.
    """
    weight = _get_weights_for_pruning(layer)
    return np.mean(np.abs(weight), axis=tuple(range(weight.ndim - 1)))


def smallest_filters(layer, pruning_rate):
    """Returns indices of the smallest filters to prune, based on the rate
    of filters to prune.
    Pruning rate must be a value between 0 and 1. For instance, 0.3 means
    that 30% of the filters will be pruned. The function then returns the
    indices of the 30% smallest filters in this layer.

    Args:
        layer (keras.Layer): the layer to prune.
        pruning_rate (float): the pruning rate (between 0 and 1).

    Returns:
        `:obj:np.ndarray`: the indices of the filters to prune.
    """
    l1_norms = _filter_magnitudes(layer)
    indices_sorted = np.argsort(l1_norms)
    num_filters_to_prune = int(np.round(l1_norms.size * pruning_rate))
    return indices_sorted[:num_filters_to_prune]


def prune_model(model,
                acceptance_function,
                pruning_rates=None,
                prunable_layers_policy=neural_layers,
                prunable_filters_policy=smallest_filters):
    """Prune model automatically based on an acceptance function.

    The algorithm for filter pruning is as follows:

        1. Select the first prunable layer (according to the
           ``prunable_layers_policy`` function).
        2. As long as the ``acceptance_function`` returns True, prune
           successively the layer with different pruning rates (according to
           ``pruning_rates`` and ``prunable_filters_policy``).
        3. When the current pruned model is not acceptable, the last valid
           pruning rate is selected for the final pruned model.
        4. Repeat steps 1, 2 and 3 for the next prunable layers.


    Examples:

        .. code-block:: python

            acceptable_drop = 0.05

            def evaluate(model):
                _, accuracy = model.evaluate(data, labels)
                return accuracy

            ref_accuracy = evaluate(base_model)

            def acceptance_function(pruned_model):
                # This function returns True if the pruned_model is acceptable.
                # Here, the pruned model is acceptable if the accuracy drops
                # less than 5% from the base model.

                return ref_accuracy - evaluate(pruned_model) <= acceptable_drop

            # Prune model
            pruned_model, pruning_rates = prune_model(model, acceptance_function)

    Args:
        model (keras.Model): a keras model to prune
        acceptance_function (function): a criterion function that returns True
            if the pruned model is acceptable. The signature must be
            `function(model)`.
        pruning_rates (list, optional): a list of pruning rates to test. Default
            is [0.1, 0.2, ..., 0.9].
        prunable_layers_policy (function, optional): a function returning a list
            of layers to prune in the model. The signature must be
            `function(model)`, and must return a list of prunable layer names.
            By default, all neural layers (Conv2D/SeparableConv2D/Dense/
            QuantizedConv2D/QuantizedSeparableConv2D/QuantizedDense) are
            candidates for pruning.
        prunable_filters_policy (function, optional): a function that returns
            the filters to prune in a given layer for a specific pruning rate.
            The signature must be `function(layer, pruning_rate)` and returns a
            list of indices to prune. By default, filters with the lowest
            magnitude are pruned.

    Returns:
        tuple: the pruned model and the pruning rates.
    """

    if pruning_rates is None:
        pruning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Start by creating a copy of the model where batchnorm is folded
    pruned_model = cnn2snn.transforms.sequentialize(model)
    pruned_model = cnn2snn.transforms.invert_batchnorm_pooling(pruned_model)
    pruned_model = cnn2snn.transforms.fold_batchnorm(pruned_model)

    # Get the list of prunable layers
    layers_to_prune = prunable_layers_policy(pruned_model)

    # Initialize pruning rates (no pruning)
    rates = {name: 0.0 for name in layers_to_prune}

    for name in layers_to_prune:
        layer = pruned_model.get_layer(name)
        next_pruned_model = None

        for pruning_rate in pruning_rates:
            # Get the list of prunable filters for the specified rate
            filters = prunable_filters_policy(layer, pruning_rate)
            candidate_model = delete_filters(pruned_model, layer.name, filters)

            if acceptance_function(candidate_model):
                # The candidate model passed the acceptance criteria
                next_pruned_model = candidate_model
                rates[name] = pruning_rate
            else:
                # Stop at that rate
                break

        if next_pruned_model:
            pruned_model = next_pruned_model

    return pruned_model, rates
