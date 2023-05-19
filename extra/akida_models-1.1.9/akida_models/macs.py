import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph)

from quantizeml.models import load_model
from cnn2snn import load_quantized_model


def get_flops(model):
    """
    Calculate FLOPS for a tf.keras.Model or tf.keras.Sequential model in inference mode.

    It uses tf.compat.v1.profiler under the hood.

    Args:
        model (:obj:`keras.Model`): the model to evaluate

    Returns:
        :obj:`tf.compat.v1.profiler.GraphNodeProto`: object containing the FLOPS
    """
    # Prepare a constant input to pass to the profiler
    input_shape = model.inputs[0].shape.as_list()
    input_shape[0] = 1
    x = tf.constant(tf.fill(input_shape, 1))
    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError("Calculating FLOPS is only supported for `tf.keras.Model`"
                         "and `tf.keras.Sequential` instances.")

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(x)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    return flops


def display_macs(model_path, verbose=False):
    """Displays the MACs for a keras model

    By default it displays only the total MACS.

    Args:
        model (:obj:`keras.Model`): the model to evaluate
        verbose (bool): display MACS for each operation

    """
    try:
        model = load_model(model_path)
    except ValueError:
        model = load_quantized_model(model_path)

    flops = get_flops(model)

    if verbose:
        def display_children_macs(nodes):
            for node in nodes:
                print(f"{node.name}: {node.total_float_ops / 2:e} MACS")
                display_children_macs(node.children)
        # Recursively display MACS by node (i.e. operation)
        display_children_macs(flops.children)

    # We divide FLOPS by 2 to obtain an estimate of Multiply and Accumulate (MACS)
    print(f"Total: {flops.total_float_ops / 2:e} MACS")
