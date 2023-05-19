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
 Toolset to load IBM DVS Gesture preprocessed dataset.

Brainchip information:
        - AER2.0 specifications:
            https://inivation.com/support/software/fileformat/#bit-30-12
        - AER3.1 specifications:
            https://inivation.com/support/software/fileformat/#aedat-31
"""

import struct
import os
import re
import sys
import tarfile
import urllib
import glob

import numpy as np

# Global definition for AEDATv2
EVENT_LENGTH = 8  # 1 event takes 8 bytes (64 bits)
READ_MODE = '>II'  # struct.unpack(), 4 bytes + 4 bytes
TIMESTEP = 0.000001  # timestep is 1us
XMASK = 0x00fe
XSHIFT = 1
YMASK = 0x7f00
YSHIFT = 8
PMASK = 0x1


def _parse_version_string(version_string):
    """Analyze string version from AEDAT file.

    Args:
        version_string (str): value contains file version.

    Returns:
        int: document major version.
    """
    v2 = re.compile('.*#!AER-DAT2.*')
    v3 = re.compile('.*#!AER-DAT3.*')
    v4 = re.compile('.*#!AER-DAT4.*')
    if v2.match(str(version_string)):
        return 2
    if v3.match(str(version_string)):
        return 3
    if v4.match(str(version_string)):
        return 4
    raise ValueError("Unrecognized or unsupported version: %s" %
                     str(version_string).strip(),
                     file=sys.stderr)


def _skip_ascii_header(file, version):
    """Going through header without reading data.

    Args:
        file (obj): opened AEDAT file
        version (int): AEDAT major version

    Returns:
        int: End of header bytes position.
    """
    bytes_pos = 0  # bytes position
    line = file.readline()
    bytes_pos += len(line)
    if version in [2, 3]:
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            line = file.readline()
            bytes_pos += len(line)
    elif version == 4:
        return bytes_pos
    else:
        raise ValueError("Unsupported version: %d" % version)
    return bytes_pos


def _read_aedat_v2(file, file_length, bytes_pos):
    """Read event in AEDAT v2 format.

    Args:
        file (obj): opened AEDAT file.
        file_length (int): AEDAT file length.
        bytes_pos (int): starting events header position.

    Returns:
        tuple: list of events with a tuple (x, y, polarity) and a list of
            events time.
    """

    # create data lists
    ev_times = []
    events = []

    # read events data
    file.seek(bytes_pos)
    s = file.read(EVENT_LENGTH)
    bytes_pointer = bytes_pos + EVENT_LENGTH

    while bytes_pointer <= file_length:
        addr, ts = struct.unpack(READ_MODE, s)

        # get coordinates and polarity
        x = (addr & XMASK) >> XSHIFT
        y = (addr & YMASK) >> YSHIFT
        polarity = (addr & PMASK)

        ev_times.append(ts)
        events.append((x, y, polarity))

        # move to next event
        file.seek(bytes_pointer)
        s = file.read(EVENT_LENGTH)
        bytes_pointer += EVENT_LENGTH

    return events, ev_times


def _read_event_v3(file, bytes_pos, trial_times, subj_num, light_cond):
    """Read event in AEDAT v3 format.

    Args:
        file (obj): opened AEDAT file.
        bytes_pos (int): starting events header position.
        trial_times (array): start/stop times for gestures from CSV label file.
        subj_num (int): subject id.
        light_cond (str): lighting condition.
    Returns:
        list: readed data are stored in dict and added to the returned list.
    """
    file.seek(bytes_pos)  # seek to end of header

    # init trials values and reference timestamps
    trials = []
    trial_ind = 1
    num_trials = len(trial_times)
    start_time = int(trial_times[trial_ind][1])
    stop_time = int(trial_times[trial_ind][2])

    ev_list = []
    ev_times = []
    while True:
        # events header part
        header = file.read(28)
        if not header or len(header) == 0:
            break

        # read header
        header_type = struct.unpack('H', header[0:2])[0]
        size = struct.unpack('I', header[4:8])[0]
        tsoverflow = struct.unpack('I', header[12:16])[0]
        capacity = struct.unpack('I', header[16:20])[0]

        data_length = capacity * size
        data = file.read(data_length)
        counter = 0

        if header_type == 1:
            while data[counter:counter + size]:
                # unpack data from bytes
                aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                timestamp = struct.unpack(
                    'I', data[counter + 4:counter + 8])[0] | tsoverflow << 31
                x = (aer_data >> 17) & 0x00007FFF
                y = (aer_data >> 2) & 0x00007FFF
                pol = (aer_data >> 1) & 0x00000001
                counter = counter + size

                if timestamp >= start_time:
                    if timestamp < stop_time:
                        # Collect the events as a list (coordinates and times
                        # separately)
                        ev_list.append((x, y, pol))
                        ev_times.append(timestamp)
                    else:
                        # Once stop_time is reached,
                        # Collect the event lists and accompanying details as a
                        # dictionary
                        trial = {
                            'label': trial_times[trial_ind][0],
                            'subject': subj_num,
                            'lighting': light_cond,
                            'events': np.array(ev_list, dtype=np.uint8),
                            'ev_times': np.array(ev_times) - ev_times[0]
                        }
                        trials.append(trial)
                        # Then reset other values ready to collect the next
                        # trial
                        ev_list = []
                        ev_times = []
                        trial_ind += 1
                        if trial_ind < num_trials:
                            start_time = int(trial_times[trial_ind][1])
                            if start_time < stop_time:
                                print("Error on subject %d, with %s condition" %
                                      (subj_num, light_cond))
                                raise Exception("New start time earlier than \
                                    previous block stop time")
                            stop_time = int(trial_times[trial_ind][2])
                        else:
                            return trials
        else:
            # non-polarity event packet, not implemented
            raise NotImplementedError("Non polarity event packet not supported")
    return []


def _read_aedat_v4(datafile):

    # create data lists
    ev_times = []
    events = []

    try:
        from dv import AedatFile

        with AedatFile(datafile) as f:
            np_events = np.array([],
                                 dtype=[('timestamp', '<i8'), ('x', '<i2'),
                                        ('y', '<i2'), ('polarity', 'i1'),
                                        ('_p1', '<i2'), ('_p2', 'i1')])
            for packet in f['events'].numpy():
                np_events = np.hstack([np_events, packet])

            # access information of all events by type
            timestamps, x, y, polarities = np_events['timestamp'], np_events[
                'x'], np_events['y'], np_events['polarity']
            ev_times = timestamps.tolist()
            events = np.column_stack((x, y, polarities)).tolist()
    except Exception as e:
        raise ImportError(
            f"dv package is required to read aedat v4 files: {e}") from e

    return events, ev_times


def get_gesture_data(dataset_name, datafile, labelfile=None):
    """Creates gesture data dictionary from aedat file.

    Args:
        dataset_name (str): dataset name (dvs_gesture or samsung_handy).
        datafile (str): absolute path to aedat file.
        labelfile (int): absolute path to related label file.

    Returns:
        dict: loaded data stored in a dict with gesture id, subject id,
            lighting condition, events and events time.

    Notes
    -----
    The method returns a dict for dvs_handy dataset and a list of dict for
    dvs_gesture dataset.
    """

    data = None
    if os.path.isfile(datafile):
        if dataset_name == "dvs_gesture":
            with open(labelfile, encoding='UTF-8') as f:
                # Get class labels and start/stop times from the labels.csv file
                trial_times = [line.split(',') for line in f]

            # Get subject id and lightning condition from filename
            subj_num = int(os.path.basename(datafile)[4:6])
            lighting_condition = os.path.basename(datafile)[7:-6]
        elif dataset_name == "samsung_handy":
            subj = int(os.path.basename(datafile)[5:7])
            label = int(os.path.basename(datafile)[8:10])
            lighting_condition = 'fluorescent'

        with open(datafile, 'rb') as aef:
            version = _parse_version_string(aef.readline())
            aef.seek(0, 0)  # rewind
            bytes_pos = _skip_ascii_header(aef, version)
            if version == 2:
                # get file length
                statinfo = os.stat(datafile)
                file_length = statinfo.st_size

                events, ev_times = _read_aedat_v2(aef, file_length, bytes_pos)
                data = {
                    'label': label,
                    'subject': subj,
                    'lighting': lighting_condition,
                    'events': np.array(events, dtype=np.uint8),
                    'ev_times': np.array(ev_times) - ev_times[0]
                }
            elif version == 3:
                data = _read_event_v3(aef, bytes_pos, trial_times, subj_num,
                                      lighting_condition)
            elif version == 4:
                events, ev_times = _read_aedat_v4(datafile)
                data = {
                    'label': label,
                    'subject': subj,
                    'lighting': lighting_condition,
                    'events': np.array(events, dtype=np.uint16),
                    'ev_times': np.array(ev_times) - ev_times[0]
                }
            else:
                raise ValueError("Unsupported version: %d" % version)

    return data


def generate_dataset(dataset_name, datadir, subject_list):
    """Generates data set for a subject id list.

    Args:
        dataset_name (str): dataset name (dvs_gesture or samsung_handy).
        datadir (str): filepath where aedat files are stored.
        subject_list (list): subject id list to compose the train/test set.

    Returns:
        list: dataset filled with all events data.
    """
    dataset = []
    if dataset_name == "dvs_gesture":
        for subj in subject_list:
            datafile = glob.glob(
                os.path.join(datadir, 'DvsGesture',
                             'user' + '{0:0=2d}'.format(subj) + '*.aedat'))
            labelfile = glob.glob(
                os.path.join(datadir, 'DvsGesture',
                             'user' + '{0:0=2d}'.format(subj) + '*.csv'))
            datafile.sort()
            labelfile.sort()

            for idx, val in enumerate(datafile):
                dataset_tmp = get_gesture_data(dataset_name, val,
                                               labelfile[idx])
                for data_dict in dataset_tmp:
                    dataset.append(data_dict)
    elif dataset_name == "samsung_handy":
        for subj in subject_list:
            datafile = glob.glob(
                os.path.join(datadir,
                             'GestS' + '{0:0=2d}'.format(subj) + '*.aedat*'))
            datafile.sort()
            for val in datafile:
                data = get_gesture_data(dataset_name, val)
                if data:
                    dataset.append(data)
    else:
        raise ValueError("Unsupported dataset: %s" % dataset_name)
    return dataset


def download_and_extract_dataset(data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads data from given url and unpacks it into a directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
        return
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        try:
            filepath, _ = urllib.request.urlretrieve(data_url, filepath,
                                                     _progress)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download URL: {data_url} to folder: "
                f"{filepath}\nPlease make sure you have enough free"
                f" space and an internet connection") from e
        print()
        statinfo = os.stat(filepath)
        print(f"Successfully downloaded {filename} ({statinfo.st_size} bytes)")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(dest_directory)
