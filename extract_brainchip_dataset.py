import numpy as np
import pickle
import argparse, math, shutil, os, json, time
from PIL import Image
from extract_data

parser = argparse.ArgumentParser(description='Edge Impulse => Brainchip YOLOv2')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
# parser.add_argument('--anchors', type=int, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())
with open(os.path.join(args.data_directory, 'Y_split_test.npy'), 'r') as f:
    Y_test = json.loads(f.read())

# This is the part that causes the issue. The X_train does not have enough data to populate 3 variables.
image_width, image_height, image_channels = list(X_train.shape[1:])

# delete any previously existing versions of this data
out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with Brainchip\'s YOLOv2 training pipeline')

def current_ms():
    return round(time.time() * 1000)

total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0


def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    # List to hold all the dictionaries holding data for each image
    all_data = []

    for ix in range(0, len(X)):
        # Create one data dictionary for each image
        data = {"boxes": []}

        raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        # Save images to directory to pass to YoloV2
        images_dir = os.path.join(out_dir, category, 'images')
        labels_dir = os.path.join(out_dir, category, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # where is this image data coming from? - the raw data is being used to convert to image and then store it
        im = Image.fromarray(raw_img_data)
        image_path = os.path.join(images_dir, 'image' + str(ix).zfill(5) + '.jpg')
        im.save(image_path)

        # Add image path and image shape to data dictionary
        data['image_path'] = image_path
        data['image_shape'] = (image_width, image_height, image_channels)

        labels_text = []

        # All the labels found in one image
        for l in labels:
            if (l['label'] > class_count):
                class_count = l['label']

            # Dimensions of one bounding box:
            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            # Class x_center y_center width height
            x_center = (x + (w / 2)) / image_width
            y_center = (y + (h / 2)) / image_height
            width = w / image_width
            height = h / image_height

            labels_text.append(str(l['label'] - 1) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))

            # Save label and centroid data of each image in its own file
            with open(os.path.join(labels_dir, 'image' + str(ix).zfill(5) + '.txt'), 'w') as f:
                f.write('\n'.join(labels_text))

            # Dimensions of one bounding box as required by Brainchip
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h

            # Create another dictionary to hold bounding box data and labels
            box = {}

            box['label'] = l['label']
            box['x1'] = int(round(float(x1)))
            box['y1'] = int(round(float(y1)))
            box['x2'] = int(round(float(x2)))
            box['y2'] = int(round(float(y2)))

            # After the above 5 pieces of info are added to the key, append to "box"
            # starts again if more than one bounding box + label in one image
            if len(box) == 5:
              data["boxes"].append(box)

        # If there are any bounding boxes located in the image, append to all_data, otherwise skip append  
        if len(data["boxes"]) != 0:
            all_data.append(data)

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()


    return(all_data)


all_train_data = convert(X=X_train, Y=Y_train, category='train')
all_valid_data = convert(X=X_test, Y=Y_test, category='valid')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with Brainchip\'s YOLOv2 OK')
print('')

labels = []
for c in range(0, class_count):
    labels.append("'class" + str(c) + "'")
labels = ', '.join(labels)


data_yaml = """
train: """ + os.path.join(os.path.abspath(out_dir), 'train', 'images') + """
val: """ + os.path.join(os.path.abspath(out_dir), 'valid', 'images') + """

nc: """ + str(class_count) + """
names: [""" + labels + """]
"""

with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

# Combine all preprocessed data into one dataset to dump in pickle file for akida training
dataset = []
dataset.append(all_train_data)
dataset.append(all_valid_data)
dataset.append(labels)

with open('voc_preprocessed.pkl', 'wb') as file:
    pickle.dump(dataset, file)

########################################################################################################
# ******************************************************************************
# MIT License
#
# Copyright (c) 2017 Ngoc Anh Huynh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ******************************************************************************
"""
This module provides a method to generate YOLO anchors from dataset annotations.
"""

import random
# import numpy as np


def _iou(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def _avg_iou(anns, centroids):
    n, _ = anns.shape
    s = 0.

    for i in range(anns.shape[0]):
        s += max(_iou(anns[i], centroids))

    return s / n


def _run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num) * (-1)
    iteration = 0

    indices = [random.randrange(ann_dims.shape[0]) for _ in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - _iou(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()


def generate_anchors(annotations_data, num_anchors=5, grid_size=(7, 7)):
    """ Creates anchors by clustering dimensions of the ground truth boxes
    from the training dataset.

    Args:
        annotations_data (dict): dictionnary of preprocessed VOC data
        num_anchors (int, optional): number of anchors
        grid_size (tuple, optional): size of the YOLO grid

    Returns:
        list: the computed anchors
    """
    annotation_dims = []

    # run k_mean to find the anchors
    for item in annotations_data:
        cell_w = item['image_shape'][0] / grid_size[0]
        cell_h = item['image_shape'][1] / grid_size[1]

        for box in item['boxes']:
            relative_w = float(box['x2'] - box['x1']) / cell_w
            relative_h = float(box['y2'] - box['y1']) / cell_h
            annotation_dims.append(tuple(map(float, (relative_w, relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = _run_kmeans(annotation_dims, num_anchors)
    print('\nAverage IOU for', num_anchors, 'anchors:',
          '%0.2f' % _avg_iou(annotation_dims, centroids))
    anchors = np.sort(np.round(centroids, 5), 0).tolist()
    print('Anchors: ', anchors)
    return anchors

########################################################################################################

# Generate anchors for Brainchip's akida YOLOv2 training using the YOLO toolkit


# from akida_models.detection.generate_anchors import generate_anchors

# num_of_anchors = args.anchors
num_of_anchors = 5
grid_size = (7, 7)

anchors = generate_anchors(all_train_data, num_of_anchors, grid_size)

with open('voc_anchors.pkl', 'wb') as file:
    pickle.dump(anchors, file)



# # https://github.com/TexasInstruments/edgeai-yolov5/blob/master/models/hub/yolov5s6.yaml
# yolo_spec = """# parameters
# nc: """ + str(class_count) + """  # number of classes
# depth_multiple: 0.33  # model depth multiple
# width_multiple: 0.50  # layer channel multiple
# anchors:
#   - [ 19,27,  44,40,  38,94 ]  # P3/8
#   - [ 96,68,  86,152,  180,137 ]  # P4/16
#   - [ 140,301,  303,264,  238,542 ]  # P5/32
#   - [ 436,615,  739,380,  925,792 ]  # P6/64

# # YOLOv5 backbone
# backbone:
#   # [from, number, module, args]
#   [ [ -1, 1, Focus, [ 64, 3 ] ],  # 0-P1/2
#     [ -1, 1, Conv, [ 128, 3, 2 ] ],  # 1-P2/4
#     [ -1, 3, C3, [ 128 ] ],
#     [ -1, 1, Conv, [ 256, 3, 2 ] ],  # 3-P3/8
#     [ -1, 9, C3, [ 256 ] ],
#     [ -1, 1, Conv, [ 512, 3, 2 ] ],  # 5-P4/16
#     [ -1, 9, C3, [ 512 ] ],
#     [ -1, 1, Conv, [ 768, 3, 2 ] ],  # 7-P5/32
#     [ -1, 3, C3, [ 768 ] ],
#     [ -1, 1, Conv, [ 1024, 3, 2 ] ],  # 9-P6/64
#     [ -1, 1, SPP, [ 1024, [ 3, 5, 7 ] ] ],
#     [ -1, 3, C3, [ 1024, False ] ],  # 11
#   ]

# # YOLOv5 head
# head:
#   [ [ -1, 1, Conv, [ 768, 1, 1 ] ],
#     [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
#     [ [ -1, 8 ], 1, Concat, [ 1 ] ],  # cat backbone P5
#     [ -1, 3, C3, [ 768, False ] ],  # 15

#     [ -1, 1, Conv, [ 512, 1, 1 ] ],
#     [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
#     [ [ -1, 6 ], 1, Concat, [ 1 ] ],  # cat backbone P4
#     [ -1, 3, C3, [ 512, False ] ],  # 19

#     [ -1, 1, Conv, [ 256, 1, 1 ] ],
#     [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
#     [ [ -1, 4 ], 1, Concat, [ 1 ] ],  # cat backbone P3
#     [ -1, 3, C3, [ 256, False ] ],  # 23 (P3/8-small)

#     [ -1, 1, Conv, [ 256, 3, 2 ] ],
#     [ [ -1, 20 ], 1, Concat, [ 1 ] ],  # cat head P4
#     [ -1, 3, C3, [ 512, False ] ],  # 26 (P4/16-medium)

#     [ -1, 1, Conv, [ 512, 3, 2 ] ],
#     [ [ -1, 16 ], 1, Concat, [ 1 ] ],  # cat head P5
#     [ -1, 3, C3, [ 768, False ] ],  # 29 (P5/32-large)

#     [ -1, 1, Conv, [ 768, 3, 2 ] ],
#     [ [ -1, 12 ], 1, Concat, [ 1 ] ],  # cat head P6
#     [ -1, 3, C3, [ 1024, False ] ],  # 32 (P6/64-xlarge)

#     [ [ 23, 26, 29, 32 ], 1, Detect, [ nc, anchors ] ],  # Detect(P3, P4, P5, P6)
#   ]
# """

# with open(os.path.join(out_dir, 'yolov5s.yaml'), 'w') as f:
#     f.write(yolo_spec)
