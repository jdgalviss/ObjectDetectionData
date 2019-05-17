#!/bin/bash
if [ -d /usr/src/app/data/train_data ]; then
    rm -r /usr/src/app/data/train_data/*
fi
if [ ! -d /usr/src/app/data/train_data ]; then
    mkdir /usr/src/app/data/train_data
fi
export PIPELINE_CONFIG_PATH=/usr/src/app/data/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco.config
export MODEL_DIR=/usr/src/app/data/train_data
cd /usr/src/app/models/research
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --train_dir=${MODEL_DIR} \