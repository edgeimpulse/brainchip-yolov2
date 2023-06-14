## syntax = docker/dockerfile:experimental
FROM ubuntu:22.04

## Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

## Makes Poetry behave more like npm, with deps installed inside a .venv folder
## See https://python-poetry.org/docs/configuration/#virtualenvsin-project
#ENV POETRY_VIRTUALENVS_IN_PROJECT=true

## System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip build-essential
RUN apt install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc 

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

#WORKDIR /usr/local/lib/python3.10/dist-packages/akida_models/detection
#RUN rm map_evaluation.py
#COPY map_evaluation.py ./

WORKDIR /scripts

## Grab akidanet imagenet pretrained weights to be used for yolov2
RUN wget http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50.h5

WORKDIR /scripts

## Copy the normal files (e.g. run.sh and the extract_dataset scripts, etc. in)
COPY . ./


#RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENTRYPOINT ["/bin/bash", "run.sh"]
