#!/usr/bin/env bash

PREFIX=../
UNT_Aerial_Dataset=../data/UNT_Aerial_Dataset/
DATA_DIR=./data/


# Create links for UNT_Aerial_Dataset.
#ls -l ${UNT_Aerial_Dataset}
#
#if [ -e ${DATA_DIR} ]
#then
#   echo "data directory already exists."
#else
#   echo "data directory does not exists, and create this directory."
#   mkdir -p ${DATA_DIR}
#fi
#
#ln -s ${PREFIX}${UNT_Aerial_Dataset} ${DATA_DIR}UNT_Aerial_Dataset


# Create links for Pascal_VOC2007.
PASCAL_VOC2007=../data/VOC2007

ls -l ${PASCAL_VOC2007}

if [ -e ${DATA_DIR}VOC2007 ]
then
    echo "The link already exists."
else
    echo "The link doest not exists."
    ln -s ${PREFIX}${PASCAL_VOC2007} ${DATA_DIR}VOC2007
fi
