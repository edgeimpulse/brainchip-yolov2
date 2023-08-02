#####################################################################
# 6.2 Show predictions for a random image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

import argparse, pickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape
from akida_models.detection.processing import load_image, preprocess_image, decode_output

from cnn2snn import load_quantized_model, convert


parser = argparse.ArgumentParser(description='Brainchip Akida EI YOLOv2 Metric Evaluation')
parser.add_argument('--grid_size', type=int, required=True)
parser.add_argument('--num_anchors', type=int, required=True)
parser.add_argument('--classes', type=int, required=True)
parser.add_argument('--anchors_path', type=str, required=True)

args = parser.parse_args()

classes = args.classes
grid_size = (args.grid_size, args.grid_size)
num_anchors = args.num_anchors

with open(args.anchors_path, 'rb') as handle:
        anchors = pickle.load(handle)

# Load the pretrained model along with anchors
pretrained_model, anchors = load_quantized_model("akidanet_yolo_trained_iq8_wq4_aq4.h5"), anchors

with open('preprocessed_data.pkl', 'rb') as handle:
        all_data, val_data, labels = pickle.load(handle)

# Define the final reshape and build the model
output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes), name="YOLO_output")(pretrained_model.output)
model_keras = Model(pretrained_model.input, output)

# Rebuild a model without the last layer
compatible_model = Model(model_keras.input, model_keras.layers[-2].output)



model_akida = convert(compatible_model)




#------------------------------------------------------------
# with open("preprocessed_anchors.pkl", 'rb') as handle:
#         anchors = pickle.load(handle)

# with open('preprocessed_data.pkl', 'rb') as handle:
#         all_data, val_data, labels = pickle.load(handle)

# print("Loading model...")
# model_akida = Model("converted_akida_model.fbz")
# model_akida.summary()


#------------------------------------------------------------
# Take a random test image
i = np.random.randint(len(val_data))

input_shape = model_akida.layers[0].input_dims

# Load the image
raw_image = load_image(val_data[i]['image_path'])

# Keep the original image size for later bounding boxes rescaling
raw_height, raw_width, _ = raw_image.shape

# Pre-process the image
image = preprocess_image(raw_image, input_shape)
input_image = image[np.newaxis, :].astype(np.uint8)

# Call evaluate on the image
pots = model_akida.predict(input_image)[0]

# Reshape the potentials to prepare for decoding
h, w, c = pots.shape
pots = pots.reshape((h, w, len(anchors), 4 + 1 + len(labels)))

# Decode potentials into bounding boxes
raw_boxes = decode_output(pots, anchors, len(labels))

# Rescale boxes to the original image size
pred_boxes = np.array([[
    box.x1 * raw_width, box.y1 * raw_height, box.x2 * raw_width,
    box.y2 * raw_height,
    box.get_label(),
    box.get_score()
] for box in raw_boxes])


# fig = plt.figure(num='YOLOv2 object detection by Akida runtime')
# ax = fig.subplots(1)
# img_plot = ax.imshow(np.zeros(raw_image.shape, dtype=np.uint8))
# img_plot.set_data(raw_image)

# for box in pred_boxes:
#     rect = patches.Rectangle((box[0], box[1]),
#                              box[2] - box[0],
#                              box[3] - box[1],
#                              linewidth=1,
#                              edgecolor='r',
#                              facecolor='none')
#     ax.add_patch(rect)
#     class_score = ax.text(box[0],
#                           box[1] - 5,
#                           f"{labels[int(box[4])]} - {box[5]:.2f}",
#                           color='red')

# plt.axis('off')
# plt.show()