"""
Imports models.
"""

from .dvs.model_convtiny_gesture import (convtiny_dvs_gesture,
                                         convtiny_gesture_pretrained)
from .dvs.model_convtiny_handy import (convtiny_dvs_handy,
                                       convtiny_handy_samsung_pretrained)
from .imagenet.model_mobilenet import (mobilenet_imagenet,
                                       mobilenet_imagenet_pretrained)
from .imagenet.model_vgg import (vgg_imagenet, vgg_imagenet_pretrained)
from .imagenet.model_mobilenet_edge import (mobilenet_edge_imagenet,
                                            mobilenet_edge_imagenet_pretrained)
from .imagenet.model_akidanet import (
    akidanet_imagenet, akidanet_imagenet_pretrained,
    akidanet_cats_vs_dogs_pretrained, akidanet_imagenette_pretrained,
    akidanet_faceidentification_pretrained,
    akidanet_faceverification_pretrained, akidanet_melanoma_pretrained,
    akidanet_odir5k_pretrained, akidanet_retinal_oct_pretrained,
    akidanet_ecg_pretrained, akidanet_plantvillage_pretrained,
    akidanet_cifar10_pretrained, akidanet_vww_pretrained)
from .imagenet.model_akidanet_edge import (
    akidanet_edge_imagenet, akidanet_edge_imagenet_pretrained,
    akidanet_faceidentification_edge_pretrained)
from .imagenet.model_akidanet18 import akidanet18_imagenet, akidanet18_imagenet_pretrained
from .kws.model_ds_cnn import ds_cnn_kws, ds_cnn_kws_pretrained
from .modelnet40.model_pointnet_plus import (pointnet_plus_modelnet40,
                                             pointnet_plus_modelnet40_pretrained
                                             )
from .utk_face.model_vgg import vgg_utk_face, vgg_utk_face_pretrained
from .detection.model_yolo import (yolo_base, yolo_widerface_pretrained,
                                   yolo_voc_pretrained)
from .centernet.model_centernet import centernet_base
from .cwru.model_convtiny import convtiny_cwru, convtiny_cwru_pretrained
from .mnist.model_gxnor import gxnor_mnist, gxnor_mnist_pretrained
from .transformers.model_vit import (
    vit_imagenet, vit_ti16, bc_vit_ti16, bc_vit_ti16_imagenet_pretrained, vit_s16, vit_s32, vit_b16,
    vit_b32, vit_l16, vit_l32)
from .transformers.model_deit import (
    deit_imagenet, deit_ti16, bc_deit_ti16, bc_deit_dist_ti16_imagenet_pretrained,
    deit_s16, deit_b16)
from .portrait128.model_akida_unet import akida_unet_portrait128

from .calibrable import calibration_required
from .gamma_constraint import add_gamma_constraint
from .filter_pruning import delete_filters, prune_model
from .renaming import rename_model_layers
from .utils import fetch_file
