# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# 2018/12/03 -- The last time.
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import pascalvoc_2007
#from datasets import unt_aerial

datasets_map = {
    'pascalvoc_2007': pascalvoc_2007,
#    'unt_aerial': unt_aerial
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """
    Given a dataset name and a split_name returns a Dataset.

    :param name: String, the name of the dataset.
    :param split_name: A train/test split name.
    :param dataset_dir: The directory where the dataset files are stored.
    :param file_pattern: The file pattern to use for matching the dataset source files.
    :param reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    :return: A `Dataset` class.
    :raises: ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError("Name of dataset unknown %s" % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)
