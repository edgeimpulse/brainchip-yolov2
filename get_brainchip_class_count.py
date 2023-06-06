import argparse, math, shutil, os, json
import numpy as np

parser = argparse.ArgumentParser(description='Edge Impulse get number of classes for Brainchip training pipeline')
parser.add_argument('--data-directory', type=str, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())

class_count = 0

for ix in range(0, len(Y_train)):
    labels = Y_train[ix]['boundingBoxes']

    for l in labels:
        if (l['label'] > class_count):
            class_count = l['label']

print(class_count)
