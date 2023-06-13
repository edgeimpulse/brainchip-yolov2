import argparse, math, shutil, os, json
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Check contents of pickle file')
parser.add_argument('--pfile', type=str, required=True)

args = parser.parse_args()

with open(args.pfile, 'rb') as contents:
    data = pickle.load(contents)

for i in range(len(data)):
    print(data[0][i])
    print(" ")


import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('akidanet_yolo_base.h5')

# Show the model architecture
new_model.summary()


# print(" ")

# print(data[0][0]['boxes'][0]['label'])

# # Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
# with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
#     Y_train = json.loads(f.read())

# class_count = 0

# for ix in range(0, len(Y_train)):
#     labels = Y_train[ix]['boundingBoxes']

#     for l in labels:
#         if (l['label'] > class_count):
#             class_count = l['label']

# print(class_count)
