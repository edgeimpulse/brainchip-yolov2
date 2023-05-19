#!/usr/bin/env python
# ******************************************************************************
# Copyright 2021 Brainchip Holdings Ltd.
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
Preprocessing tools for face recognition: allows to detect faces and their
associated landmarks and align the faces before processing.

Based on:
    - MTCNN https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf
        title: Joint face detection and alignment using multitask cascaded
            convolutional networks.
        author: Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016)
    - matlab cp2tform python implementation to mimic SphereFace original code:
        https://github.com/clcarwin/sphereface_pytorch/blob/master/matlab_cp2tform.py
"""

import cv2
import numpy as np

from mtcnn import MTCNN
from numpy.linalg import inv, norm, lstsq


def _find_non_reflective_similarity(uv, xy):
    """
    Find Non-reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Args:
        uv (np.array): source points each row is a pair of coordinates (x, y)
        xy (np.array): each row is a pair of inverse-transformed

    Returns:
        np.array: transform matrix from uv to xy
    """

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    # We know that X * r = U
    r, _, _, _ = lstsq(X, U, rcond=None)
    r = np.squeeze(r)

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T


def _get_similarity_transform(src_pts, dst_pts):
    """
    Find Reflective Similarity Transform Matrix which can be directly used by
    cv2.warpAffine().

    Args:
        src_pts (np.array): source points
        dst_pts (np.array): destination points

    Returns:
        np.array: transform matrix from src_pts to dst_pts
    """

    def tformfwd(trans, uv):
        """ Apply affine transform 'trans' to uv
        """
        uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
        xy = np.dot(uv, trans)
        return xy[:, 0:-1]

    # Solve for trans1
    trans1 = _find_non_reflective_similarity(src_pts, dst_pts)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = dst_pts
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r = _find_non_reflective_similarity(src_pts, xyR)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, src_pts)
    norm1 = norm(xy1 - dst_pts)

    xy2 = tformfwd(trans2, src_pts)
    norm2 = norm(xy2 - dst_pts)

    if norm1 <= norm2:
        cv2_trans = trans1[:, 0:2].T
    else:
        cv2_trans = trans2[:, 0:2].T

    return cv2_trans


def preprocess_face(image_path, dst_shape=(96, 112)):
    """ Preprocess an image for face recognition by detecting the face and face
    landmarks in it before aligning the face and croping to the given size.

    Args:
        image_path (str): file path to the image

    Returns:
        np.array: the preprocessed image
    """

    # Define preprocessing constant
    ref_points = np.array([(30.2946, 51.6963), (65.5318, 51.5014),
                           (48.0252, 71.7366), (33.5493, 92.3655),
                           (62.7299, 92.2041)]).astype(np.float32)

    # Read image file
    base_image = cv2.imread(image_path)

    # Build MTCNN detector and detect faces
    detector = MTCNN()
    landmarks = detector.detect_faces(base_image)

    # Return None if no face detected
    if len(landmarks) == 0:
        return None

    # Only consider first face keypoints
    detected_points = np.array(list(landmarks[0]["keypoints"].values())).astype(
        np.float32)

    # Map keypoints to reference and apply transformation
    transform = _get_similarity_transform(detected_points, ref_points)
    return cv2.warpAffine(base_image, transform, dst_shape)
