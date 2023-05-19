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
 DVS DataGenerator class.

 This script is used in dvs_train.py to generate images from DVS data from NPY
    files. It's mandatory because Tensorflow doesn't handle events data.
"""

import random
import numpy as np


class DVS_DataGenerator:
    """ Data generator for handling event-based camera data.

    Args:
        dataset (array): array of data stored in a dict.
        subpacket_length (int, optional): subpacket length (milliseconds).
            Defaults to 100.
        subpackets_per_packet (int, optional): number of sub-packets (~ time
            bins) to concatenate to generate packets. Defaults to 1.
        camera_dims (tuple of ints, optional): dimensions of the raw data from
            the camera. Defaults to (128, 128, 2).
        include_polarity (bool, optional): keep event polarity. If False, On and
            Off events will be treated as if identical. Defaults to True.
        allow_duplicate_events (bool, optional): allow events to be
            superimposed. If False, only a single event will be allowed per
            pixel for each timebin (subpacket) and polarity. Defaults to False.
        downscale (int, optional): downscaling of inputs, implemented as integer
            division of incoming coordinate values. Defaults to 1.

    Returns:
        tuple: two numpy arrays with events and labels.
    """

    def __init__(self,
                 dataset,
                 subpacket_length=1000,
                 subpackets_per_packet=1,
                 camera_dims=(128, 128, 2),
                 include_polarity=True,
                 allow_duplicate_events=False,
                 downscale=1):
        # Convert milliseconds to microseconds
        self.subpacket_length = subpacket_length * 1000
        self.subpackets_per_packet = subpackets_per_packet
        self.downscale = downscale
        if include_polarity:
            self.camera_dims = (camera_dims[0],) + (camera_dims[1],) + (2,)
        else:
            self.camera_dims = (camera_dims[0],) + (camera_dims[1],) + (1,)
        self.include_polarity = include_polarity
        self.sample_dims = (self.camera_dims[0] // downscale,
                            self.camera_dims[1] // downscale,
                            self.camera_dims[2] * subpackets_per_packet)
        self.allow_duplicate_events = allow_duplicate_events
        self.dataset = dataset

        self.subject_list = []
        trial_durations = []
        label_offset = int(self.dataset[0]['label'])
        for trial in self.dataset:
            subj = int(trial['subject'])
            if subj not in self.subject_list:
                self.subject_list.append(subj)
            trial['label'] = int(trial['label']) - label_offset
            trial_durations.append(trial['ev_times'][-1])
        self.trial_durations = trial_durations

    def get_random_samples(self,
                           class_list,
                           class_key=None,
                           samples_per_trial=5,
                           stop_time=None):
        """ Generates [samples_per_trial] samples from each subject/lighting
        condition/class.

        Args:
            class_list (list): classes selected for training
            class_key (list, optional): keys of selected classes. Defaults to
             None.
            samples_per_trial (int, optional): number of times that every sample
             have to be repeated. Defaults to 5.
            stop_time (int, optional): samples ending point. Defaults to None.

        Returns:
            tuple: events and labels
        """
        if class_key is None:
            class_key = list(range(len(class_list)))
        outevents = []
        outlabels = []
        sample_windows = []
        for _, trial in enumerate(self.dataset):
            local_stop_time = stop_time
            if int(trial['label']) in class_list:
                if local_stop_time is None:
                    local_stop_time = trial['ev_times'][-1]
                elif local_stop_time > trial['ev_times'][-1]:
                    print('ERROR: requested stop time later than the end of \
                         the trial')
                    print('Ending samples at ' + str(trial['ev_times'][-1]))
                    local_stop_time = trial['ev_times'][-1]
                latest_start_time = local_stop_time - \
                    (self.subpacket_length * self.subpackets_per_packet)

                if latest_start_time < 0:
                    print('ERROR: Sample durations requested is longer \
                        than the available data for this trial')
                    continue
                latest_start_point = np.searchsorted(np.squeeze(
                    trial['ev_times']),
                    latest_start_time,
                    side='right') - 1

                if latest_start_point < 0:
                    print("ERROR: Latest start point isn't valid")
                    continue

                # Build samples from multiple packets
                for _ in range(samples_per_trial):
                    sample_evs = np.zeros((0, 3), dtype=np.uint8)
                    start_point = np.random.randint(latest_start_point)
                    sample_start_point = start_point
                    sample_start_time = trial['ev_times'][start_point]
                    for packet in range(self.subpackets_per_packet):
                        # Set packet end point relative to start point
                        # Convention here is EXCLUSIVE end_point, for clarity
                        # in python
                        end_time = sample_start_time + \
                            self.subpacket_length * (packet + 1)
                        end_point = np.searchsorted(np.squeeze(
                            trial['ev_times']),
                            end_time,
                            side='right')
                        # Get the actual events
                        ev_array = np.copy(trial['events'][
                            np.squeeze(start_point):np.squeeze(end_point), :])
                        if self.downscale > 1:
                            ev_array[:, 0] = ev_array[:, 0] // self.downscale
                            ev_array[:, 1] = ev_array[:, 1] // self.downscale

                        # We use the events feature dimension to encode the
                        # temporal order of packets
                        if not self.include_polarity:
                            ev_array[:, 2] = 0
                        ev_array[:, 2] += self.camera_dims[2] * packet
                        sample_evs = np.concatenate((sample_evs, ev_array))

                        # Update start point for the next packet
                        start_point = end_point
                    outevents.append(sample_evs)
                    outlabels.append(class_key[class_list.index(
                        int(trial['label']))])
                    sample_windows.append(end_point - sample_start_point)

        mean_events = np.mean(np.array(sample_windows))
        # mean_duration is in msecs
        mean_duration = self.subpacket_length / 1000 * self.subpackets_per_packet
        # mean_ev_rate stores events per second
        mean_ev_rate = mean_events / mean_duration * 1000
        print('DVS event rate: ' + str(np.round(mean_ev_rate)) + ' events/sec')

        outlabels = np.array(outlabels)
        return outevents, outlabels

    def get_continuous_sample(self,
                              subj,
                              class_label,
                              class_key=None,
                              start_time=0,
                              stop_time=None,
                              overlapping_samples=True):
        """ Generates series of temporally contiguous samples from a single
        subject+class.

        Args:
            subj (int): subject id
            class_label (int): label of the class
            class_key (int, optional): key of the class. Defaults to None.
            start_time (int, optional): samples starting point. Defaults to 0.
            stop_time (int, optional): samples ending point. Defaults to None.
            overlapping_samples(bool, optional): samples overlapping. Defaults
                to True.

        Returns:
            tuple: events, labels, plot events
        """
        if class_key is None:
            class_key = class_label
        outevents = []
        outlabels = []
        plotevents = []
        sample_windows = []
        for _, trial in enumerate(self.dataset):
            local_stop_time = stop_time
            if (int(trial['label']) == class_label) and (int(trial['subject'])
                                                         == subj):
                if local_stop_time is None:
                    local_stop_time = trial['ev_times'][-1]
                elif local_stop_time > trial['ev_times'][-1]:
                    print('ERROR: requested stop time later than the end of \
                         the trial')
                    print('Ending samples at ' + str(trial['ev_times'][-1]))
                    local_stop_time = trial['ev_times'][-1]
                latest_start_time = local_stop_time - \
                    (self.subpacket_length * self.subpackets_per_packet)
                if latest_start_time < 0:
                    print('ERROR: Sample durations requested is longer \
                        than the available data for this trial (or stop \
                        time is set too early)')
                    continue
                latest_start_point = np.searchsorted(np.squeeze(
                    trial['ev_times']),
                    latest_start_time,
                    side='right') - 1

                # Build samples from multiple packets
                if start_time > 0:
                    sample_start_point = np.searchsorted(np.squeeze(
                        trial['ev_times']),
                        start_time,
                        side='left')
                else:
                    sample_start_point = 0
                sample_start_time = start_time
                while sample_start_point < latest_start_point:
                    sample_evs = np.zeros((0, 3), dtype=np.uint8)
                    plot_evs = np.zeros((0, 3), dtype=np.int)
                    start_point = sample_start_point
                    for packet in range(self.subpackets_per_packet):
                        # Set packet end point relative to start point
                        # Convention here is EXCLUSIVE end_point, for clarity
                        # in python
                        end_time = sample_start_time + \
                            self.subpacket_length * (packet + 1)
                        end_point = np.searchsorted(np.squeeze(
                            trial['ev_times']),
                            end_time,
                            side='right')

                        # Get the actual events
                        plot_array = np.copy(trial['events'][
                            np.squeeze(start_point):np.squeeze(end_point), :])
                        plot_evs = np.concatenate((plot_evs, plot_array))

                        ev_array = np.copy(trial['events'][
                            np.squeeze(start_point):np.squeeze(end_point), :])
                        if self.downscale > 1:
                            ev_array[:, 0] = ev_array[:, 0] // self.downscale
                            ev_array[:, 1] = ev_array[:, 1] // self.downscale
                        if not self.include_polarity:
                            ev_array[:, 2] = 0
                        ev_array[:, 2] += self.camera_dims[2] * packet
                        sample_evs = np.concatenate((sample_evs, ev_array))

                        # Update start point for the next packet
                        start_point = end_point
                    outevents.append(sample_evs)
                    outlabels.append(class_key)
                    plotevents.append(plot_evs)
                    sample_windows.append(end_point - sample_start_point)
                    if overlapping_samples:
                        sample_start_time += self.subpacket_length
                    else:
                        sample_start_time += self.subpacket_length * self.subpackets_per_packet
                    sample_start_point = np.searchsorted(np.squeeze(
                        trial['ev_times']),
                        sample_start_time,
                        side='right')
        outlabels = np.array(outlabels)
        return outevents, outlabels, plotevents

    def events2matrix(self, event_packet):
        """ Transforms a DVS aer event into a numpy array.

        Args:
            event_packet (numpy.ndarray): input events packet

        Returns:
            numpy.ndarray: matrix of events
        """
        if self.allow_duplicate_events:
            out_mat = np.zeros((self.sample_dims))
            for ind in range(event_packet.shape[0]):
                out_mat[event_packet[ind, 1], event_packet[ind, 0],
                        event_packet[ind, 2]] += 1
        else:
            length = np.prod(np.array(self.sample_dims, dtype=int))
            out_mat = np.zeros((length, 1))
            indices = np.ravel_multi_index(
                (event_packet[:, 1], event_packet[:, 0], event_packet[:, 2]),
                (self.sample_dims))
            out_mat[indices] = 1
            out_mat = np.reshape(out_mat, (self.sample_dims))
        return out_mat.astype(np.float16)

    def samples_evs2images(self, samples_events):
        """ Transforms a list of DVS aer events into a matrix suitable for
        TensorFlow training.

        Args:
            samples_events (list): list of DVS events

        Returns:
            np.ndarray: events as an image array
        """
        outmat = []
        for sample in samples_events:
            sample_mat = self.events2matrix(sample)
            outmat.append(sample_mat)

        out_arr = np.array(outmat)

        if self.allow_duplicate_events:
            out_arr = np.clip(out_arr, 0, 15)

        return out_arr

    def augment_data(self,
                     samples_events,
                     horizontal_flip=False,
                     vertical_flip=False,
                     polarity_flip=False,
                     rotation_range=0,
                     width_shift_range=1.0,
                     height_shift_range=1.0):
        """ Performs data augmentations on sample events.

        Args:
            samples_events (list): list of DVS events to process
            horizontal_flip (bool, optional): enable horizontal flip. Defaults
             to False.
            vertical_flip (bool, optional): enable vertical flip. Defaults to
             False.
            polarity_flip (bool, optional): enable polarity flip. Defaults to
             False.
            rotation_range (int, optional): rotation angle range in degrees.
             Defaults to 0.
            width_shift_range (float, optional): range of width shift. Defaults
             to 1.0.
            height_shift_range (float, optional): range of height shift.
             Defaults to 1.0.

        Returns:
            list: augmented events
        """
        outevents = []
        for sample in samples_events:
            samp = np.copy(sample).astype(np.float)
            # To make subsequent transforms easier, centre matrix around (0, 0)
            samp[:, 0] -= self.sample_dims[1] // 2
            samp[:, 1] -= self.sample_dims[0] // 2

            if horizontal_flip and random.random() > 0.5:
                samp[:, 0] = -samp[:, 0]
            if vertical_flip and random.random() > 0.5:
                samp[:, 1] = -samp[:, 1]
            if polarity_flip and random.random() > 0.5:
                # shift all odd numbers by -2
                samp[(samp[:, 2] % 2).astype(np.bool), 2] -= 2
                samp[:, 2] += 1
            if rotation_range > 0:
                # degrees to radians
                theta = random.uniform(-rotation_range,
                                       rotation_range) * np.pi / 180
                rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta),
                                        np.cos(theta)]])
                samp[:, :2] = np.dot(rot_matrix, samp[:, :2].T).T
            if width_shift_range != 1.0:
                scale = random.uniform(1, width_shift_range)
                if random.random() > 0.5:
                    # Scale Up
                    samp[:, 0] = samp[:, 0] * scale
                else:
                    # Scale down
                    samp[:, 0] = samp[:, 0] / scale
            if height_shift_range != 1.0:
                scale = random.uniform(1, width_shift_range)
                if random.random() > 0.5:
                    # Scale Up
                    samp[:, 1] = samp[:, 1] * scale
                else:
                    # Scale down
                    samp[:, 1] = samp[:, 1] / scale
            samp[:, 0] += self.sample_dims[1] // 2
            samp[:, 1] += self.sample_dims[0] // 2
            samp = np.round(samp)
            out_of_range = np.where(
                np.logical_or(
                    np.logical_or(samp[:, 0] >= self.sample_dims[1],
                                  samp[:, 0] < 0),
                    np.logical_or(samp[:, 1] >= self.sample_dims[0],
                                  samp[:, 1] < 0)))
            samp = np.delete(samp, out_of_range, axis=0)
            outevents.append(samp.astype(np.uint8))
        return outevents
