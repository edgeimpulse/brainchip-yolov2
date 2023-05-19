#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

python3 -c "import numpy as np; print(np.__version__)"
echo " "

# # wget http://data.brainchip.com/dataset-mirror/voc/voc_test_car_person.tar.gz
echo "Running akida_models create model"
akida_models create -s yolo_akidanet_voc.h5 yolo_base -c 2 -bw akidanet_imagenet_alpha_50.h5

echo "Running preprocess"
python3 01.preprocess.py

yolo_train train -d voc_preprocessed.pkl -m yolo_akidanet_voc.h5 -ap voc_anchors.pkl -e 1 -fb 1conv -s yolo_akidanet_voc.h5
cnn2snn quantize -m yolo_akidanet_voc.h5 -iq 8 -wq 4 -aq 4
yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5
cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 1 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5
yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5
cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 1 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5

python3 04.metrics.py

# python3 01.preprocess.py
# python3 plot_4_transfer_learning.py

# POSITIONAL_ARGS=()

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     --epochs) # e.g. 50
#       EPOCHS="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     --learning-rate) # e.g. 0.01
#       LEARNING_RATE="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     --data-directory) # e.g. 0.2
#       DATA_DIRECTORY="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     --out-directory) # e.g. (96,96,3)
#       OUT_DIRECTORY="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     *)
#       POSITIONAL_ARGS+=("$1") # save positional arg
#       shift # past argument
#       ;;
#   esac
# done

# if [ -z "$EPOCHS" ]; then
#     echo "Missing --epochs"
#     exit 1
# fi
# if [ -z "$LEARNING_RATE" ]; then
#     echo "Missing --learning-rate"
#     exit 1
# fi
# if [ -z "$DATA_DIRECTORY" ]; then
#     echo "Missing --data-directory"
#     exit 1
# fi
# if [ -z "$OUT_DIRECTORY" ]; then
#     echo "Missing --out-directory"
#     exit 1
# fi

# OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
# DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

# IMAGE_SIZE=$(python3 get_image_size.py --data-directory "$DATA_DIRECTORY")

# # convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv5 understands)
# python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# cd /app/yolov5
# # train:
# #     --freeze 10 - freeze the bottom layers of the network
# #     --workers 0 - as this otherwise requires a larger /dev/shm than we have on Edge Impulse prod,
# #                   there's probably a workaround for this, but we need to check with infra.
# python3 -u train.py --img $IMAGE_SIZE \
#     --freeze 10 \
#     --epochs $EPOCHS \
#     --data /tmp/data/data.yaml \
#     --weights /app/yolov5n.pt \
#     --name yolov5_results \
#     --cache \
#     --workers 0
# echo "Training complete"
# echo ""

# mkdir -p $OUT_DIRECTORY

# # export as onnx
# echo "Converting to ONNX..."
# python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --img $IMAGE_SIZE --include onnx
# cp runs/train/yolov5_results/weights/last.onnx $OUT_DIRECTORY/model.onnx
# echo "Converting to ONNX OK"
# echo ""

# # export as f32
# echo "Converting to TensorFlow Lite model (fp16)..."
# python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --img $IMAGE_SIZE --include saved_model tflite --keras
# cp runs/train/yolov5_results/weights/last-f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         p16.tflite $OUT_DIRECTORY/model.tflite
# # ZIP up and copy the saved model too
# cd runs/train/yolov5_results/weights/last_saved_model
# zip -r -X ./saved_model.zip . > /dev/null
# cp ./saved_model.zip $OUT_DIRECTORY/saved_model.zip
# cd /app/yolov5
# echo "Converting to TensorFlow Lite model (fp16) OK"
# echo ""

# export as i8 (skipping for now as it outputs a uint8 input, not an int8 - which the Studio won't handle)
# echo "Converting to TensorFlow Lite model (int8)..."
# python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --data /tmp/data/data.yaml --img $IMAGE_SIZE --include tflite --int8
# cp runs/train/yolov5_results/weights/last-int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
# echo "Converting to TensorFlow Lite model (int8) OK"
# echo ""

