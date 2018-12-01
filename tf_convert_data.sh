#!/usr/bin/env bash

UNT_AERIAL_DATASET=./data/UNT_Aerial_Dataset/train/
PASCAL_VOC=./data/VOC2007/train/

ls -l ${UNT_AERIAL_DATASET}

python3 tf_convert_data.py  \
    --dataset_name=unt_aerial    \
    --dataset_dir=${UNT_AERIAL_DATASET}    \
    --output_name=unt_aerial_dataset_train \
    --output_dir=/tmp/tfrecord1