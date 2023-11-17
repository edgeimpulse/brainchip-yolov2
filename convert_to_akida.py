######################################################################
# 6. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 6.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Check model compatibility before akida conversion
#

import argparse, pickle
from cnn2snn import load_quantized_model, check_model_compatibility, convert
from timeit import default_timer as timer
from akida_models.detection.map_evaluation import MapEvaluation
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Reshape
from conversion import convert_to_float32
import os, zipfile


parser = argparse.ArgumentParser(description='Brainchip Akida EI YOLOv2 Metric Evaluation')
parser.add_argument('--grid-size', type=int, required=True)
parser.add_argument('--num-anchors', type=int, required=True)
parser.add_argument('--classes', type=int, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--anchors_path', type=str, required=True)
parser.add_argument('--preprocessed_data_path', type=str, required=True)

args = parser.parse_args()

classes = args.classes
grid_size = (args.grid_size, args.grid_size)
num_anchors = args.num_anchors

# with open(os.path.join(args.out_directory, "akida_yolov2_anchors.pkl"), 'rb') as handle:
#     anchors = pickle.load(handle)

with open(args.anchors_path, 'rb') as handle:
        anchors = pickle.load(handle)

# Load the pretrained model along with anchors
pretrained_model, anchors = load_quantized_model("akidanet_yolo_trained_iq8_wq4_aq4.h5"), anchors
float32_model = load_model("akidanet_yolo_trained.h5")

# with open(os.path.join(args.out_directory, 'preprocessed_data.pkl'), 'rb') as handle:
#     all_data, val_data, labels = pickle.load(handle)

with open(args.preprocessed_data_path, 'rb') as handle:
        all_data, val_data, labels = pickle.load(handle)


# Define the final reshape and build the model
output_keras = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes), name="YOLO_output")(pretrained_model.output)
model_keras = Model(pretrained_model.input, output_keras)

output_keras_32 = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes), name="YOLO_output")(float32_model.output)
model_keras_f32 = Model(float32_model.input, output_keras_32)

# Check model compatibility for model_keras
print("Checking model compatibility for model_keras: \n")
compatibility_keras = check_model_compatibility(model_keras, False)

# Check model compatibility for model_keras_32
print("Checking model compatibility for model_keras_f32: \n")
compatibility_keras_32 = check_model_compatibility(model_keras_f32, False)

######################################################################
# The last YOLO_output layer that was added for splitting channels into values
# for each box must be removed before akida conversion.

# Rebuild a model without the last layer
compatible_model = Model(model_keras.input, model_keras.layers[-2].output)
model_keras_f32 = Model(model_keras_f32.input, model_keras_f32.layers[-2].output)

######################################################################
# When converting to an Akida model, we just need to pass the Keras model
# and the input scaling that was used during training to `cnn2snn.convert
# <../../api_reference/cnn2snn_apis.html#convert>`_. In YOLO
# `preprocess_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.preprocess_image>`_
# function, images are zero centered and normalized between [-1, 1] hence the
# scaling values.

model_akida = convert(compatible_model)
model_akida.summary()

model_akida.save(os.path.join(args.out_directory, "akida_model.fbz"))

h5_path = os.path.join(args.out_directory, "model.h5")
model_keras_f32.save(h5_path, save_format='h5')

with zipfile.ZipFile(h5_path + '.zip', "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(h5_path, os.path.basename(h5_path))
os.remove(h5_path)

print("Model Saved as akida_model.fbz and model.h5.zip")

saved_model_dir = os.path.join(args.out_directory, "saved_model")
model_keras_f32.save(saved_model_dir, save_format='tf')

MODEL_INPUT_SHAPE = model_keras.input.shape[1:]
print("Model Input Shape: ", MODEL_INPUT_SHAPE)

# Create tflite files (f32 / i8)
convert_to_float32(model_keras_f32, args.out_directory, MODEL_INPUT_SHAPE, 'model.tflite')
print("Model saved as 'model.tflite'. \n")


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
num_images = 100
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


print("----------------------------------------------------------------")
print('mAP: {:.4f}'.format(mAP_ak))
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')
print("----------------------------------------------------------------")
