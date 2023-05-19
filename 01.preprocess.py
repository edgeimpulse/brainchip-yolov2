######################################################################
# 2. Preprocessing tools
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As this example focuses on car and person detection only, a subset of VOC has
# been prepared with test images from VOC2007 that contains at least one
# of the occurence of the two classes. Just like the VOC dataset, the subset
# contains an image folder, an annotation folder and a text file listing the
# file names of interest.
#
# The `YOLO toolkit <../../api_reference/akida_models_apis.html#yolo-toolkit>`_
# offers several methods to prepare data for processing, see
# `load_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.load_image>`_,
# `preprocess_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.preprocess_image>`_
# or `parse_voc_annotations <../../api_reference/akida_models_apis.html#akida_models.detection.processing.parse_voc_annotations>`_.
#
#

import os
import pickle
from tensorflow.keras.utils import get_file
from akida_models.detection.processing import parse_voc_annotations

cwd = os.getcwd()
# Download validation set from Brainchip data server
# data_path = get_file(
#     "voc_test_car_person.tar.gz",
#     "http://data.brainchip.com/dataset-mirror/voc/voc_test_car_person.tar.gz",
#     # cache_subdir='datasets/voc',
#     cache_subdir=os.path.join(cwd, 'new/datasets/voc'),
#     extract=True)

# data_path = '/home/parallels/yolov2/new/datasets/voc'
data_path = os.path.join(cwd, 'voc')
print("Data Path: ", data_path)
# data_dir = os.path.dirname(data_path)
data_dir = data_path
print("Data Dir: ", data_dir)
annotations_folder = os.path.join(data_dir, 'voc_test_car_person', 'Annotations')
print("Annotations Folder: ", annotations_folder)
image_folder = os.path.join(data_dir, 'voc_test_car_person', 'JPEGImages')
print("Image Folder: ", image_folder)
# file_path = os.path.join(data_dir, 'voc_test_car_person', 'test_car_person.txt')
# print("File Path: ", file_path)
labels = ['car', 'person']

train_file = os.path.join(data_dir, 'voc_test_car_person', 'train_car_person.txt')
print("File Path Train: ", train_file)

val_file = os.path.join(data_dir, 'voc_test_car_person', 'val_car_person.txt')
print("File Path Val: ", val_file)

# val_data = parse_voc_annotations(annotations_folder, image_folder, file_path, labels)
tri_data = parse_voc_annotations(annotations_folder, image_folder, train_file, labels)
val_data = parse_voc_annotations(annotations_folder, image_folder, val_file, labels)

print("Loaded VOC2007 train data for car and person classes: "f"{len(tri_data)} images.")
print("Loaded VOC2007 validation data for car and person classes: "f"{len(val_data)} images.")

data = []
data.append(tri_data)
data.append(val_data)
data.append(labels)

with open('voc_preprocessed.pkl', 'wb') as file:
    pickle.dump(data, file)

######################################################################
# Anchors can also be computed easily using YOLO toolkit.
#
# .. Note:: The following code is given as an example. In a real use case
#           scenario, anchors are computed on the training dataset.

from akida_models.detection.generate_anchors import generate_anchors

num_anchors = 5
grid_size = (7, 7)
anchors_example = generate_anchors(val_data, num_anchors, grid_size)

with open('voc_anchors.pkl', 'wb') as file:
    pickle.dump(anchors_example, file)

with open('voc_preprocessed.pkl', 'rb') as handle:
    data = pickle.load(handle)
    print(len(data))
    train_data, valid_data, labels_2 = data

print(len(train_data))
print(len(valid_data))
print(labels_2)
