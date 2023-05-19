######################################################################
# 4. Training
# ~~~~~~~~~~~
#
# As the YOLO model relies on Brainchip AkidaNet/ImageNet network, it is
# possible to perform transfer learning from ImageNet pretrained weights when
# training a YOLO model. See the `PlantVillage transfer learning example
# <plot_4_transfer_learning.html>`_ for a detail explanation on transfer
# learning principles.
#
# When using transfer learning for YOLO training, we advise to proceed in
# several steps that include model calibration:
#

# STEP ONE:
# * instantiate the `yolo_base` model and load AkidaNet/ImageNet pretrained
#   float weights,
#
# .. code-block:: bash
#
#       akida_models create -s yolo_akidanet_voc.h5 yolo_base --classes 2 \
#                --base_weights akidanet_imagenet_224_alpha_50.h5
##		HERE CREATE MODEL WITH TOTAL NUMBER OF CLASSES AVAILABLE IN YOLOV5

# STEP TWO:
# * freeze the AkidaNet layers and perform training,
#
# .. code-block:: bash
#
#       yolo_train train -d voc_preprocessed.pkl -m yolo_akidanet_voc.h5 \
#           -ap voc_anchors.pkl -e 25 -fb 1conv -s yolo_akidanet_voc.h5

# STEP THREE:
# * quantize the network, create data for calibration and calibrate,
#
# .. code-block:: bash
#
#       cnn2snn quantize -m yolo_akidanet_voc.h5 -iq 8 -wq 4 -aq 4
#       yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz \
#           -m yolo_akidanet_voc_iq8_wq4_aq4.h5
#
#       cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 500 -lr 1e-3 \
#           -m yolo_akidanet_voc_iq8_wq4_aq4.h5

# STEP FOUR:
# * tune the model to recover accuracy.
#
# .. code-block:: bash
#
#       yolo_train tune -d voc_preprocessed.pkl \
#           -m yolo_akidanet_voc_iq8_wq4_aq4_adaround_calibrated.h5 -ap voc_anchors.pkl \
#           -e 10 -s yolo_akidanet_voc_iq8_wq4_aq4.h5


# IMPORTANT NOTES:
# .. Note::
#
#       - ``voc_anchors.pkl`` is obtained saving the output of the
#         `generate_anchors` call to a pickle file,

#       - ``voc_preprocessed.pkl`` is obtained saving training data, validation
#         data (obtained using `parse_voc_annotations`) and labels list (i.e
#         ["car", "person"]) into a pickle file.
#
#
# Even if transfer learning should be the preferred way to train a YOLO model, it
# has been observed that for some datasets training all layers from scratch
# gives better results. That is the case for our `YOLO WiderFace model
# <../../api_reference/akida_models_apis.html#akida_models.yolo_widerface_pretrained>`_
# to detect faces. In such a case, the training pipeline to follow is described
# in the `typical training scenario
# <../../user_guide/cnn2snn.html#typical-training-scenario>`_.
#