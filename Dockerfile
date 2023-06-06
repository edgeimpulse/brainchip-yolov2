## syntax = docker/dockerfile:experimental
FROM ubuntu:22.04

## Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

## Makes Poetry behave more like npm, with deps installed inside a .venv folder
## See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

## System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip

## Install TensorFlow and other dependencies for akida libraries
COPY install_tensorflow.sh install_tensorflow.sh
RUN yes | ./install_tensorflow.sh

#COPY install_addons.sh install_addons.sh
#RUN ./install_addons.sh
#    rm install_tensorflow.sh

## Local dependencies
COPY requirements-mini.txt ./

RUN pip3 install -r requirements-mini.txt 

RUN pip3 install akida
RUN pip3 install cnn2snn
RUN pip3 install akida-models
RUN pip3 install -r requirements-mini.txt
RUN pip3 install protobuf==3.19.0
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /scripts

## Grab akidanet imagenet pretrained weights to be used for yolov2
RUN wget http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50.h5

WORKDIR /scripts

## Copy the normal files (e.g. run.sh and the extract_dataset scripts, etc. in)
COPY . ./


#RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
#RUN akida_models create -s yolo_akidanet_voc.h5 yolo_base -c 2 -bw akidanet_imagenet_alpha_50.h5
#RUN echo "Model created"

#RUN python3 01.preprocess.py

#RUN yolo_train train -d voc_preprocessed.pkl -m yolo_akidanet_voc.h5 -ap voc_anchors.pkl -e 1 -fb 1conv -s yolo_akidanet_voc.h5
#RUN cnn2snn quantize -m yolo_akidanet_voc.h5 -iq 8 -wq 4 -aq 4
#RUN yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5
#RUN cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 1 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5
#RUN yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5
#RUN cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 1 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5

ENTRYPOINT ["/bin/bash", "run.sh"]
