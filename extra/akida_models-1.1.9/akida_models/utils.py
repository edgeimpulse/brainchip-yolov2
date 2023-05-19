#!/usr/bin/env python
# ******************************************************************************
# Copyright 2022 Brainchip Holdings Ltd.
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
Utilities for akida_models package.
"""

import os
import urllib
import time

from six.moves.urllib.parse import urlsplit

import tensorflow as tf
from keras.utils import io_utils
from keras.utils.data_utils import validate_file, _extract_archive
from keras.utils.generic_utils import Progbar
from keras.callbacks import TensorBoard

from cnn2snn import load_quantized_model
from quantizeml.models import load_model as qml_load_model


def fetch_file(origin, fname=None, file_hash=None, cache_subdir="datasets", extract=False,
               cache_dir=None):
    """ Downloads a file from a URL if it is not already in the cache.

    Reimplements `keras.utils.get_file` without raising an error when detecting a file_hash
    mismatch (it will just re-download the model).

    Args:
        origin (str): original URL of the file.
        fname (str, optional): name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If `None`, the name of the file at
            `origin` will be used. Defaults to None.
        file_hash (str, optional): the expected hash string of the file after download. Defaults to
            None.
        cache_subdir (str, optional): subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is specified the file will be saved at that
            location. Defaults to 'datasets'.
        extract (bool, optional): True tries extracting the file as an Archive, like tar or zip.
            Defaults to False.
        cache_dir (str, optional): location to store cached files, when None it defaults to the
            default directory `~/.keras/`. Defaults to None.

    Returns:
        str: path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras")

    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fname = io_utils.path_to_string(fname)
    if not fname:
        fname = os.path.basename(urlsplit(origin).path)
        if not fname:
            raise ValueError(f"Can't parse the file name from the origin provided: '{origin}'."
                             "Please specify the `fname` as the input param.")

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found, verify integrity if a hash was provided.
        if file_hash is not None and not validate_file(fpath, file_hash):
            io_utils.print_msg("A local file was found, but it seems to be incomplete or outdated"
                               "because the file hash does not match the original value of "
                               f"{file_hash} so we will re-download the data.")
            download = True
    else:
        download = True

    if download:
        io_utils.print_msg(f"Downloading data from {origin}.")

        class DLProgbar:
            """Manage progress bar state for use in urlretrieve."""

            def __init__(self):
                self.progbar = None
                self.finished = False

            def __call__(self, block_num, block_size, total_size):
                if not self.progbar:
                    if total_size == -1:
                        total_size = None
                    self.progbar = Progbar(total_size)
                current = block_num * block_size
                if current < total_size:
                    self.progbar.update(current)
                elif not self.finished:
                    self.progbar.update(self.progbar.target)
                    self.finished = True

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urllib.request.urlretrieve(origin, fpath, DLProgbar())
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if extract:
        _extract_archive(fpath, datadir)

    return fpath


def load_model(model_path):
    """Combine the cnn2snn.load_quantized_model and quantizeml.load_model

    Args:
        model_path (str): model path

    Returns:
        keras.Model: the load model
    """
    try:
        model = load_quantized_model(model_path)
    except Exception:
        try:
            model = qml_load_model(model_path)
        except Exception as e:
            raise e.__class__('Cannot load provided model.')
    return model


def get_tensorboard_callback(out_dir, histogram_freq=1, prefix=''):
    """Build a Tensorboard call, pointing to the output directory

    Args:
        out_dir (str): parent directory of the folder to create
        histogram_freq (int, optional): frequency to export logs. Defaults to 1.
        prefix (str, optional): prefix name. Defaults to ''.
    """
    def _create_log_dir(out_dir, prefix=''):
        if len(prefix) != 0 and not prefix.endswith('_'):
            prefix += '_'
        base_name = prefix + time.strftime('%Y_%m_%d.%H_%M_%S', time.localtime())
        log_dir = os.path.join(out_dir, base_name)

        print('Saving tensorboard and checkpoint information to:', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print('Directory', log_dir, 'created ...')
        else:
            print('Directory', log_dir, 'already exists ...')
        return log_dir

    log_dir = _create_log_dir(out_dir, prefix)
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    return TensorBoard(log_dir=log_dir,
                       histogram_freq=histogram_freq,
                       update_freq='epoch',
                       write_graph=False,
                       profile_batch=0)
