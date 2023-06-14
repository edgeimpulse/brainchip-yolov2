######################################################################
# 5. Performance
# ~~~~~~~~~~~~~~
#
# The model zoo also contains an `helper method
# <../../api_reference/akida_models_apis.html#akida_models.yolo_voc_pretrained>`_
# that allows to create a YOLO model for VOC and load pretrained weights for the
# car and person detection task and the corresponding anchors. The anchors are
# used to interpret the model outputs.
#
# The metric used to evaluate YOLO is the mean average precision (mAP) which is
# the percentage of correct prediction and is given for an intersection over
# union (IoU) ratio. Scores in this example are given for the standard IoU of
# 0.5 meaning that a detection is considered valid if the intersection over
# union ratio with its ground truth equivalent is above 0.5.
#
#  .. Note:: A call to `evaluate_map <../../api_reference/akida_models_apis.html#akida_models.detection.map_evaluation.MapEvaluation.evaluate_map>`_
#            will preprocess the images, make the call to ``Model.predict`` and
#            use `decode_output <../../api_reference/akida_models_apis.html#akida_models.detection.processing.decode_output>`__
#            before computing precision for all classes.
#
# Reported performanced for all training steps are as follows:
#
# +------------+-----------+------------------+-------------+
# |            |   Float   | 8/4/4 Calibrated | 8/4/4 Tuned |
# +============+===========+==================+=============+
# | Global mAP |  38.38 %  | 32.88 %          | 38.83 %     |
# +------------+-----------+------------------+-------------+

import argparse, pickle
from timeit import default_timer as timer
from akida_models.detection.map_evaluation import MapEvaluation
from cnn2snn import load_quantized_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape

parser = argparse.ArgumentParser(description='Brainchip Akida EI YOLOv2 Metric Evaluation')
parser.add_argument('--grid-size', type=int, required=True)
parser.add_argument('--num-anchors', type=int, required=True)
parser.add_argument('--classes', type=int, required=True)

args = parser.parse_args()

classes = args.classes
grid_size = (args.grid_size, args.grid_size)
# grid_size = (7, 7)
num_anchors = args.num_anchors
# num_anchors = 5

# anchors_path = "preprocessed_anchors.pkl"
# model_path   = "akidanet_yolo_trained_iq8_wq4_aq4.h5"

with open("preprocessed_anchors.pkl", 'rb') as handle:
        anchors = pickle.load(handle)

# Load the pretrained model along with anchors
pretrained_model, anchors = load_quantized_model("akidanet_yolo_trained_iq8_wq4_aq4.h5"), anchors

with open('preprocessed_data.pkl', 'rb') as handle:
        all_data, val_data, labels = pickle.load(handle)


# Define the final reshape and build the model
output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes), name="YOLO_output")(pretrained_model.output)
model_keras = Model(pretrained_model.input, output)


# Create the mAP evaluator object
num_images = 100
map_evaluator = MapEvaluation(model_keras, val_data[:num_images], labels, anchors)


# Compute the scores for all validation images
start = timer()
mAP, average_precisions = map_evaluator.evaluate_map()
end = timer()


for label, average_precision in average_precisions.items():
    print(labels[label], '{:.4f}'.format(average_precision))

print("----------------------------------------------------------------")
print('mAP: {:.4f}'.format(mAP))
print(f'Keras inference on {num_images} images took {end-start:.2f}s.\n')
print("----------------------------------------------------------------")
