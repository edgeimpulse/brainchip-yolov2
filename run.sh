#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# add arguments in the studio: number of anchors, grid size

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

IMAGE_SIZE=$(python3 get_brainchip_image_size.py --data-directory "$DATA_DIRECTORY")
CLASSES=$(python3 get_brainchip_class_count.py --data-directory "$DATA_DIRECTORY")

########### REMOVE WHEN TESTING IN STUDIO!! ####################

# ## The `-d` test command option see if FILE exists and is a directory ##
# if [ -d $OUT_DIRECTORY ]; then
#     rm -rf $OUT_DIRECTORY
#     exit 1
# fi

# mkdir "$OUT_DIRECTORY"

ls -r
echo " "
echo "$OUT_DIRECTORY"

################### TILL HERE ##################################

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv2 understands)
echo " "
echo "Extracting dataset for YOLOv2 and generating anchors"
echo " "
python3 -u extract_brainchip_dataset.py \
        --data-directory $DATA_DIRECTORY \
        --out-directory $OUT_DIRECTORY \
        --temp-directory /tmp/data

# start Akida training pipeline
echo " "
echo "Creating Akidanet YOLO model"
akida_models create --save_model akidanet_yolo_base.h5 yolo_base \
        --classes $CLASSES \
        --base_weights akidanet_imagenet_224_alpha_50.h5


echo " "
echo "Initiating first training cycle"
yolo_train train --data $OUT_DIRECTORY/preprocessed_data.pkl \
        --model akidanet_yolo_base.h5 \
        --anchors_path $OUT_DIRECTORY/akida_yolov2_anchors.pkl \
        --epochs $EPOCHS \
        --freeze_before 1conv \
        --savemodel akidanet_yolo_trained.h5

echo " "
echo "Initiating quantization of trained model"
cnn2snn quantize --model akidanet_yolo_trained.h5 \
        --input_weight_quantization 8 \
        --weight_quantization 4 \
        --activ_quantization 4

echo " "
echo "Initiating extraction of calibration data"
yolo_train extract --data $OUT_DIRECTORY/preprocessed_data.pkl \
        --anchors_path $OUT_DIRECTORY/akida_yolov2_anchors.pkl \
        --batch_size 1024 \
        --out_file sample_data.npz \
        --model akidanet_yolo_trained_iq8_wq4_aq4.h5   

echo " "
echo "Initiating calibration of quantized model"
cnn2snn calibrate adaround --samples sample_data.npz \
        --batch_size 128 \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --model akidanet_yolo_trained_iq8_wq4_aq4.h5

# *trained-model-filename*_iq8_wq4_aq4_adaround_calibrated.h5 <-- This file is generated using the name of the original trained model filename
echo " "
echo "Initiating final tuning of quantized model"
yolo_train tune --data $OUT_DIRECTORY/preprocessed_data.pkl \
        --model akidanet_yolo_trained_iq8_wq4_aq4_adaround_calibrated.h5 \
        --anchors_path $OUT_DIRECTORY/akida_yolov2_anchors.pkl \
        --epochs $EPOCHS \
        --savemodel akidanet_yolo_trained_iq8_wq4_aq4.h5

echo " "
echo "Getting metrics for quantized model"
python3 get_brainchip_metrics.py --grid_size 7 \
        --num_anchors 5 \
        --classes $CLASSES \
        --anchors_path $OUT_DIRECTORY/akida_yolov2_anchors.pkl \
        --preprocessed_data_path $OUT_DIRECTORY/preprocessed_data.pkl

echo " "
echo "Initiating conversion of quantized model to Akida"
python3 convert_to_akida.py --grid-size 7 \
        --num-anchors 5 \
        --classes $CLASSES \
        --out-directory $OUT_DIRECTORY \
        --anchors_path $OUT_DIRECTORY/akida_yolov2_anchors.pkl \
        --preprocessed_data_path $OUT_DIRECTORY/preprocessed_data.pkl


# # echo " "
# # echo "Running Predictions on model"
# # python3 run_akida_predictions.py --grid_size 7 \
# #         --num_anchors 5 \
# #         --classes $CLASSES 
