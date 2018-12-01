# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Qi 2018-09-29
"""
Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory `JPEGImages`. Similarly, bounding box annotations are supposed to be
stored int he `Annotation directory`.

This Tensorflow script converts the training and evaluation data into a sharded
data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records.
Each record within the TFRecord file is a serialized Example proto. The Example
proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixel
    image/width: integer, image width in pixel
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always `JPEG`

    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index
    image/object/bbox/label_text: list of string descriptions

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""

import os
import sys
import random

import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

UNT_AERIAL_LABELS = {
    'none': (0, 'Background'),
    'person': (1, 'Person'),
    'bicycle': (2, 'Vehicle'),
    'car': (3, 'Vehicle'),
    'truck': (4, 'Vehicle'),
    'bus': (5, 'Vehicle'),
    'train': (6, 'Vehicle'),
    'boat': (7, 'Vehicle'),
}

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'Images/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILE = 200


def _process_image(directory, name):
    """
    Process a image and annotation file.
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'     # directory should end up with /
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the xml annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    # print(filename)
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image size.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []

    for obj in root.findall('object'):
        label = obj.find('name').text               # eg. 'chair'
        labels.append(int(UNT_AERIAL_LABELS[label][0]))    # eg. 9
        labels_text.append(label.encode('ascii'))   # eg. b'chair'

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, shape, bboxes, labels, labels_text, difficult, truncated):
    """
    Build an Example proto for an image example.

    :return:
        An Example proto.
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format = b'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """
    Loads data from image and annotations files and add them to a TFRecord.

    :param dataset_dir: Dataset directory.
    :param name: Image name to add to the TFRecord.
    :param tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, shape, bboxes, labels,
                                  labels_text, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='unt_aerial_dataset_train', shuffling=False):
    """
    Runs the convertion operation.

    :param dataset_dir: The dataset directory where the dataset is stored.
    :param name: The output file name.
    :param output_dir: The output directory.
    :param shuffling: Shuffle the data or not.
    """
    if not tf.gfile.Exists(dataset_dir):
        print("The dataset_dir is invalid.")
        # tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open a new TFRecord file
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILE:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the Pscal VOC dataset!')
