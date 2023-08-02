import numpy as np
import pickle
import argparse, math, shutil, os, json, time
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse dataset => Brainchip YOLOv2 dataset')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--temp-directory', type=str, required=True)
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
if os.path.exists(args.temp_directory) and os.path.isdir(args.temp_directory):
    shutil.rmtree(args.temp_directory)

print('Converting dataset')

def current_ms():
    return round(time.time() * 1000)

# helper variables for pretty info prints
total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
converted_images = 0

class_count = 0

def convert(X, Y, category):
    global class_count, total_images, zf, converted_images
    last_printed = current_ms()

    # List to hold all the dictionaries holding data for each image
    all_data = []

    for ix in range(0, len(X)):
        # Create one data dictionary for each image
        data = {"boxes": []}

        raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        # Save images to directory to pass to YoloV2
        images_dir = os.path.join(args.temp_directory, category, 'images')
        labels_dir = os.path.join(args.temp_directory, category, 'labels')
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

            # Dimensions of one bounding box as required by Brainchip
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h

            # Create another dictionary to hold bounding box data and labels
            box = {}

            box['label'] = str(l['label'] - 1)
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

    return all_data


all_train_data = convert(X=X_train, Y=Y_train, category='train')
all_valid_data = convert(X=X_test, Y=Y_test, category='valid')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Converting dataset OK')
print('')

labels = []
for c in range(0, class_count):
    # labels.append("class" + str(c))
    labels.append(str(c))

# Combine all preprocessed data into one dataset to dump in pickle file for akida training
dataset = []
dataset.append(all_train_data)
dataset.append(all_valid_data)
dataset.append(labels)

with open('preprocessed_data.pkl', 'wb') as file:
    pickle.dump(dataset, file)

print('Generating YOLOv2 anchors...')
# Generate anchors for Brainchip's akida YOLOv2 training using the YOLO toolkit
from akida_models.detection.generate_anchors import generate_anchors

# below are default values for generate_anchors function
num_of_anchors = 5
grid_size = (7, 7)
anchors = generate_anchors(all_train_data, num_of_anchors, grid_size)

print(f"Saving anchors to {os.path.join(args.out_directory, 'akida_yolov2_anchors.pkl')}")
with open(os.path.join(args.out_directory, 'akida_yolov2_anchors.pkl'), 'wb') as file:
    pickle.dump(anchors, file)

print('Generating YOLOv2 anchors OK')
