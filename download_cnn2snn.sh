#!/bin/bash
set -e

cd /app

# Download the library
wget https://files.pythonhosted.org/packages/ff/db/dbe8709ceee1985e0a0256d5cb47cda3cae31bf2e7bd2ca5da760e4ad5fe/cnn2snn-2.2.2.tar.gz
tar -xvzf cnn2snn-2.2.2.tar.gz
rm cnn2snn-2.2.2.tar.gz

# Patch it to be compatible with TF2.7
rm cnn2snn-2.2.2/setup.py
cp cnn2snn/setup.py cnn2snn-2.2.2
rm cnn2snn-2.2.2/cnn2snn.egg-info/requires.txt
cp cnn2snn/requires.txt cnn2snn-2.2.2/cnn2snn.egg-info

