#!/usr/bin/env python
# ******************************************************************************
# Copyright 2020 Brainchip Holdings Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
akida_models main command-line interface

This package entry-point allows akida models to be instantiated and saved at a
specified location.

"""

import argparse

from .utk_face.model_vgg import vgg_utk_face
from .kws.model_ds_cnn import ds_cnn_kws
from .modelnet40.model_pointnet_plus import pointnet_plus_modelnet40
from .imagenet.model_mobilenet import mobilenet_imagenet
from .imagenet.model_vgg import vgg_imagenet
from .imagenet.model_akidanet import akidanet_imagenet
from .imagenet.model_akidanet_edge import akidanet_edge_imagenet
from .imagenet.model_akidanet18 import akidanet18_imagenet
from .imagenet.model_mobilenet_edge import mobilenet_edge_imagenet
from .detection.model_yolo import yolo_base
from .dvs.model_convtiny_handy import convtiny_dvs_handy
from .dvs.model_convtiny_gesture import convtiny_dvs_gesture
from .cwru.model_convtiny import convtiny_cwru
from .mnist.model_gxnor import gxnor_mnist
from .transformers.model_vit import vit_ti16, bc_vit_ti16
from .transformers.model_deit import deit_ti16, bc_deit_ti16
from .portrait128.model_akida_unet import akida_unet_portrait128
from .centernet.model_centernet import centernet_base
from .macs import display_macs


def add_transformers_args(parser, add_customs=False, default_norm='LN', default_act='GeLU'):
    """ Add transformers commons arguments to the given parser.

    Args:
        parser (argparse.ArgumentParser): parser to add args to
        add_customs (bool, optional): indicates if normalization, softmax,
            last normalization and activation can be customized. Defaults to
            False.
        default_norm (str, optional): when add_customs=True, default value for normalization.
            Defaults to 'LN'.
        default_act (str, optional): when add_customs=True, default value for activation. Defaults
            to 'GeLU'.
    """
    parser.add_argument("-c", "--classes", type=int, default=1000,
                        help="The number of classes, by default %(default)s.")
    parser.add_argument("-bw", "--base_weights", type=str, default=None,
                        help="Optional keras weights to load in the model, by default %(default)s.")
    if add_customs:
        norms = ['LN', 'GN1', 'BN', 'LMN']
        parser.add_argument("--norm", type=str, choices=norms, default=default_norm,
                            help="Replace normalization in model with a custom function, by \
                                  default %(default)s")
        last_norms = ['LN', 'BN']
        parser.add_argument("--last_norm", type=str, choices=last_norms,
                            default=default_norm,
                            help=f"Replace last normalization in model with a \
                                  custom function, by default {default_norm}")
        sms = ['softmax', 'softmax2']
        parser.add_argument("--softmax", type=str, choices=sms, default='softmax',
                            help="Replace softmax operation in model with custom function, by \
                                  default %(default)s")
        acts = ['GeLU', 'ReLU8', 'swish']
        parser.add_argument("--act", type=str, choices=acts, default=default_act,
                            help="Replace activation function in model with custom function, \
                                  by default %(default)s")


def add_common_args(parser, default_classes, default_alpha=None, default_img_size=None,
                    bw_required=None):
    """ Add commons arguments to the given parser.

    Args:
        parser (argparse.ArgumentParser): parser to add args to.
        default_classes (int): default number of classes.
        default_alpha (float, optional): default alpha. Defaults to None.
        default_img_size (int, optional): default image size. Defaults to None.
        bw_required (bool, optional): when base_weights is required. Defaults to None.
    """
    parser.add_argument("-c", "--classes", type=int, default=default_classes,
                        help="The number of classes, by default %(default)s.")
    if default_img_size is not None:
        img_sizes = [32, 64, 96, 128, 160, 192, 224]
        parser.add_argument("-i", "--image_size", type=int,
                            default=default_img_size, choices=img_sizes,
                            help="The square input image size, by default %(default)s.")
    if default_alpha is not None:
        parser.add_argument("-a", "--alpha", type=float, default=default_alpha,
                            help="The width of the model, by default %(default)s.")

    if bw_required is not None:
        parser.add_argument("-bw", "--base_weights", type=str, required=bw_required,
                            help="The base weights to load on model.")


def main():
    """ CLI entry point.

    Contains an argument parser with specific arguments depending on the model
    to be created. Complete arguments lists available using the -h or --help
    argument.

    """
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="action")
    c_parser = sp.add_parser("create", help="Create an akida Keras model")
    c_parser.add_argument("-s",
                          "--save_model",
                          type=str,
                          default=None,
                          help="The path/name to use to save the model")
    csp = c_parser.add_subparsers(dest="model",
                                  help="The type of model to be instantiated")
    csp.add_parser("vgg_utk_face", help="A VGG-like UTKFace model")
    csp.add_parser("convtiny_dvs_handy", help="A Convtiny DVS handy model")

    dvsh_parser = csp.add_parser("convtiny_dvs_handy",
                                 help="A Convtiny DVS model for akida")
    add_common_args(dvsh_parser, default_classes=9)

    csp.add_parser("convtiny_dvs_gesture", help="A Convtiny DVS gesture model")

    dvsg_parser = csp.add_parser("convtiny_dvs_gesture",
                                 help="A Convtiny DVS model for akida")
    add_common_args(dvsg_parser, default_classes=10)

    csp.add_parser(
        "ds_cnn_kws",
        help="A Depthwise Separable MobileNet-like model for the Keyword"
        " Spotting example")

    mb_parser = csp.add_parser("mobilenet_imagenet",
                               help="A MobileNet V1 model for akida")
    add_common_args(mb_parser, default_classes=1000, default_alpha=1.0, default_img_size=224)

    vgg_parser = csp.add_parser("vgg_imagenet", help="A VGG model for akida")
    add_common_args(vgg_parser, default_classes=1000)

    mbe_parser = csp.add_parser("mobilenet_imagenet_edge",
                                help="A MobileNet V1 model modified for akida \
                                edge learning")
    add_common_args(mbe_parser, default_classes=1000, bw_required=True)

    an_parser = csp.add_parser("akidanet_imagenet", help="An AkidaNet model")
    add_common_args(an_parser, default_classes=1000, default_img_size=224, default_alpha=1.0)

    ane_parser = csp.add_parser("akidanet_edge_imagenet",
                                help="An AkidaNet model modified for akida \
                                edge learning")
    add_common_args(ane_parser, default_classes=1000, bw_required=True)

    an18_parser = csp.add_parser("akidanet18_imagenet", help="An AkidaNet18 model")
    add_common_args(an18_parser, default_classes=1000, default_img_size=224)

    pnet_parser = csp.add_parser("pointnet_plus_modelnet40",
                                 help="A PointNet++ model for Akida")
    add_common_args(pnet_parser, default_classes=40)
    pnet_parser.add_argument("-alpha",
                             "--alpha",
                             type=float,
                             default=1.0,
                             help="Network filters multiplier (typically 0.5 , \
                             [1.0]")

    yl_parser = csp.add_parser("yolo_base", help="A YOLOv2 model for detection")
    add_common_args(yl_parser, default_classes=1, default_alpha=0.5, bw_required=False)
    yl_parser.add_argument("-na",
                           "--number_anchors",
                           type=int,
                           default=5,
                           help="The number of anchor boxes")

    csp.add_parser("convtiny_cwru", help="A Convtiny CWRU model for akida")
    csp.add_parser("gxnor_mnist", help="A GXNOR MNIST model for akida")

    akun_parser = csp.add_parser("akida_unet_portrait128",
                                 help="An Akida U-Net model for Portrait128 segmentation.")
    akun_parser.add_argument("-bw", "--base_weights", type=str, default=None,
                             help="The base AkidaNet weights to use for the encoder.")

    cn_parser = csp.add_parser("centernet", help="A CenterNet model.")
    add_common_args(cn_parser, default_classes=2, default_alpha=0.5, default_img_size=224,
                    bw_required=False)

    vit_parser = csp.add_parser("vit_ti16", help="A ViT model")
    add_transformers_args(vit_parser, add_customs=True)
    vit_parser.add_argument("-i", "--image_size", type=int, default=224,
                            choices=[224, 384], help="The square input image size")

    bc_vit_parser = csp.add_parser("bc_vit_ti16", help="A ViT, brainchip customized model")
    add_transformers_args(bc_vit_parser)

    deit_parser = csp.add_parser("deit_ti16", help="A DeiT model")
    add_transformers_args(deit_parser, add_customs=True)
    deit_parser.add_argument("-i", "--image_size", type=int, default=224,
                             choices=[160, 224], help="The square input image size")
    deit_parser.add_argument("-d", "--dist", action="store_true",
                             help="Use the model with a distilled token")

    bc_deit_parser = csp.add_parser("bc_deit_ti16", help="A DeiT, brainchip customized model")
    add_transformers_args(bc_deit_parser)
    bc_deit_parser.add_argument("-d", "--dist", action="store_true",
                                help="Use the model with a distilled token")
    bc_deit_parser.add_argument("-b", "--blocks", type=int, default=12,
                                help="Number of transformer blocks")

    macs_parser = sp.add_parser("macs", help="Get MACS for a Keras model")
    macs_parser.add_argument("-m", "--model", type=str)
    macs_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.action == "create":
        # Instantiate the wished model
        if args.model == "vgg_utk_face":
            model = vgg_utk_face()
        elif args.model == "convtiny_dvs_handy":
            model = convtiny_dvs_handy(classes=args.classes)
        elif args.model == "convtiny_dvs_gesture":
            model = convtiny_dvs_gesture(classes=args.classes)
        elif args.model == "ds_cnn_kws":
            model = ds_cnn_kws()
        elif args.model == "pointnet_plus_modelnet40":
            model = pointnet_plus_modelnet40(alpha=args.alpha,
                                             classes=args.classes)
        elif args.model == "mobilenet_imagenet":
            input_shape = (args.image_size, args.image_size, 3)
            model = mobilenet_imagenet(input_shape,
                                       alpha=args.alpha,
                                       classes=args.classes)
        elif args.model == "vgg_imagenet":
            model = vgg_imagenet(classes=args.classes)
        elif args.model == "mobilenet_imagenet_edge":
            model = mobilenet_edge_imagenet(base_model=args.base_weights,
                                            classes=args.classes)
        elif args.model == "akidanet_imagenet":
            input_shape = (args.image_size, args.image_size, 3)
            model = akidanet_imagenet(input_shape,
                                      alpha=args.alpha,
                                      classes=args.classes)
        elif args.model == "akidanet_edge_imagenet":
            model = akidanet_edge_imagenet(base_model=args.base_weights,
                                           classes=args.classes)
        elif args.model == "akidanet18_imagenet":
            input_shape = (args.image_size, args.image_size, 3)
            model = akidanet18_imagenet(input_shape, classes=args.classes)
        elif args.model == "yolo_base":
            model = yolo_base(classes=args.classes,
                              nb_box=args.number_anchors,
                              alpha=args.alpha)
            if args.base_weights is not None:
                model.load_weights(args.base_weights, by_name=True)
        elif args.model == "centernet":
            model = centernet_base(input_shape=(args.image_size, args.image_size, 3),
                                   classes=args.classes, alpha=args.alpha)
            if args.base_weights is not None:
                model.load_weights(args.base_weights, by_name=True)
        elif args.model == "convtiny_cwru":
            model = convtiny_cwru()
        elif args.model == "gxnor_mnist":
            model = gxnor_mnist()
        elif args.model == "akida_unet_portrait128":
            model = akida_unet_portrait128()
            if args.base_weights is not None:
                model.load_weights(args.base_weights, by_name=True)
        else:
            # Handle transformer models
            if args.model == "vit_ti16":
                input_shape = (args.image_size, args.image_size, 3)
                model = vit_ti16(input_shape=input_shape,
                                 classes=args.classes,
                                 norm=args.norm,
                                 last_norm=args.last_norm,
                                 softmax=args.softmax,
                                 act=args.act)
            elif args.model == "bc_vit_ti16":
                model = bc_vit_ti16(classes=args.classes)
            elif args.model == "deit_ti16":
                input_shape = (args.image_size, args.image_size, 3)
                model = deit_ti16(input_shape=input_shape,
                                  classes=args.classes,
                                  distilled=args.dist,
                                  norm=args.norm,
                                  last_norm=args.last_norm,
                                  softmax=args.softmax,
                                  act=args.act)
            elif args.model == "bc_deit_ti16":
                model = bc_deit_ti16(classes=args.classes, distilled=args.dist,
                                     num_blocks=args.blocks)
            # Load provided weights
            if args.base_weights is not None:
                model.load_weights(args.base_weights, by_name=True)
        # No need for default behaviour as the command-line parser only accepts
        # valid model types
        model_path = args.save_model
        if model_path is None:
            models_by_name = ("mobilenet_imagenet", "mobilenet_imagenet_edge", "centernet")
            model_path = f"{model.name}.h5" if args.model in models_by_name else f"{args.model}.h5"

        # If needed, add the extension
        if not model_path.endswith(".h5"):
            model_path = f"{model_path}.h5"
        model.save(model_path, include_optimizer=False)
        print(f"Model saved as {model_path}")

    elif args.action == "macs":
        display_macs(args.model, args.verbose)
