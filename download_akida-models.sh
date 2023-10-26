#!/bin/bash
set -e

cd /app

# Download the library
wget https://files.pythonhosted.org/packages/43/13/079cbe927fc0ec3224b6b620dc91b4fc5b44db514ae0b19326d5e5104c28/akida_models-1.1.3.tar.gz
tar -xvzf akida_models-1.1.3.tar.gz
rm akida_models-1.1.3.tar.gz

# Patch it to accomodate changes
rm akida_models-1.1.3/akida_models/detection/yolo_train.py
cp akida-mod/yolo_train.py akida_models-1.1.3/akida_models/detection

echo 'file replaced'

