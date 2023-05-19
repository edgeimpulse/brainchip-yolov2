######################################################################
# 3. Model architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `model zoo <../../api_reference/akida_models_apis.html#yolo>`_ contains a
# YOLO model that is built upon the `AkidaNet architecture
# <../../api_reference/akida_models_apis.html#akida_models.akidanet_imagenet>`_
# and 3 separable convolutional layers at the top for bounding box and class
# estimation followed by a final separable convolutional which is the detection
# layer. Note that for efficiency, the alpha parameter in AkidaNet (network
# width or number of filter in each layer) is set to 0.5.
#

from akida_models import yolo_base
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape

# Create a yolo model for 2 classes with 5 anchors and grid size of 7
classes = 2
num_anchors = 5
grid_size = (7, 7)

model = yolo_base(input_shape=(224, 224, 3),
                  classes=classes,
                  nb_box=num_anchors,
                  alpha=0.5)
model.summary()

######################################################################
# The model output can be reshaped to a more natural shape of:
#
#  (grid_height, grid_width, anchors_box, 4 + 1 + num_classes)
#
# where the "4 + 1" term represents the coordinates of the estimated bounding
# boxes (top left x, top left y, width and height) and a confidence score. In
# other words, the output channels are actually grouped by anchor boxes, and in
# each group one channel provides either a coordinate, a global confidence score
# or a class confidence score. This process is done automatically in the
# `decode_output <../../api_reference/akida_models_apis.html#akida_models.detection.processing.decode_output>`__
# function.

# Define a reshape output to be added to the YOLO model
output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes),
                 name="YOLO_output")(model.output)

# Build the complete model
full_model = Model(model.input, output)
full_model.output