#!/usr/bin/env bash

OUTPUT_DIR=/tmp/tfrecords1

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir ${OUTPUT_DIR}
fi

UNT_AERIAL_DATASET=./data/UNT_Aerial_Dataset/train/

ls -l ${UNT_AERIAL_DATASET}

# Data convertion for UNT Aerial Dataset.
#python3 tf_convert_data.py  \
#    --dataset_name=unt_aerial    \
#    --dataset_dir=${UNT_AERIAL_DATASET}    \
#    --output_name=unt_aerial_dataset_train \
#    --output_dir=/tmp/tfrecord1

# Data convertion for Pascal VOC 2007
PASCAL_VOC=./data/VOC2007/train/
python3 tf_convert_data.py  \
    --dataset_name=pascalvoc    \
    --dataset_dir=${PASCAL_VOC}    \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}