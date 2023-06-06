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

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv2 understands)
echo "Extracting dataset for YOLOv2"
echo " "
python3 -u extract_brainchip_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# akida_models create -save_model yolo_akidanet_voc.h5 yolo_base --classes 2 --base_weights akidanet_imagenet_224_alpha_50.h5
# yolo_train train -d voc _preprocessed.pkl -m yolo_akidanet_voc.h5 -ap voc_anchors.pkl -e 1 -fb 1conv -s yolo_akidanet_voc.h5
# cnn2snn quantize -m yolo_akidanet_voc.h5 -iq 8 -wq 4 -aq 4
# yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5
# cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 1 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5
# yolo_train tune -d voc_preprocessed.pkl -m yolo_akidanet_voc_iq8_wq4_aq4_adaround_calibrated.h5 -ap voc_anchors.pkl -e 1 -s yolo_akidanet_voc_iq8_wq4_aq4.h5



echo "Creating Akidanet YOLO model"
akida_models create --save_model yolo_akidanet_voc.h5 yolo_base \
        --classes $CLASSES \
        --base_weights akidanet_imagenet_224_alpha_50.h5


echo "Initiating first training cycle"
yolo_train train --data voc_preprocessed.pkl \
        --model yolo_akidanet_voc.h5 \
        --anchors_path voc_anchors.pkl \
        --epochs $EPOCHS \
        --freeze_before 1conv \
        --savemodel yolo_akidanet_voc.h5


cnn2snn quantize --model yolo_akidanet_voc.h5 \
        --input_weight_quantization 8 \
        --weight_quantization 4 \
        --activ_quantization 4


yolo_train extract --data voc_preprocessed.pkl \
        --anchors_path voc_anchors.pkl \
        --batch_size 1024 \
        --out_file voc_samples.npz \
        --model yolo_akidanet_voc_iq8_wq4_aq4.h5


cnn2snn calibrate adaround --samples voc_samples.npz \
        --batch_size 128 \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --model yolo_akidanet_voc_iq8_wq4_aq4.h5


# yolo_akidanet_voc_iq8_wq4_aq4_adaround_calibrated.h5 <-- where is this file coming from?
yolo_train tune --data voc_preprocessed.pkl \
        --model yolo_akidanet_voc_iq8_wq4_aq4_adaround_calibrated.h5 \
        --anchors_path voc_anchors.pkl \
        --epochs $EPOCHS \
        --savemodel yolo_akidanet_voc_iq8_wq4_aq4.h5



python3 04.metrics.py



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

