######################################################################
# 6. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 6.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Check model compatibility before akida conversion
#

from cnn2snn import check_model_compatibility

print("skipping compat check")
#compat = check_model_compatibility(model_keras, False)

######################################################################
# The last YOLO_output layer that was added for splitting channels into values
# for each box must be removed before akida conversion.

# Rebuild a model without the last layer
compatible_model = Model(model_keras.input, model_keras.layers[-2].output)

######################################################################
# When converting to an Akida model, we just need to pass the Keras model
# and the input scaling that was used during training to `cnn2snn.convert
# <../../api_reference/cnn2snn_apis.html#convert>`_. In YOLO
# `preprocess_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.preprocess_image>`_
# function, images are zero centered and normalized between [-1, 1] hence the
# scaling values.
#

from cnn2snn import convert

model_akida = convert(compatible_model)
model_akida.summary()

######################################################################
# 6.1 Check performance
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Akida model accuracy is tested on the first *n* images of the validation set.
#
# The table below summarizes the expected results:
#
# +---------+-----------+-----------+
# | #Images | Keras mAP | Akida mAP |
# +=========+===========+===========+
# | 100     |  38.80 %  |  34.26 %  |
# +---------+-----------+-----------+
# | 1000    |  40.11 %  |  39.35 %  |
# +---------+-----------+-----------+
# | 2500    |  38.83 %  |  38.85 %  |
# +---------+-----------+-----------+
#

# Create the mAP evaluator object
map_evaluator_ak = MapEvaluation(model_akida,
                                 val_data[:num_images],
                                 labels,
                                 anchors,
                                 is_keras_model=False)

# Compute the scores for all validation images
start = timer()
mAP_ak, average_precisions_ak = map_evaluator_ak.evaluate_map()
end = timer()

for label, average_precision in average_precisions_ak.items():
    print(labels[label], '{:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(mAP_ak))
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')