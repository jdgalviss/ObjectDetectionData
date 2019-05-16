#!/bin/bash
export PIPELINE_CONFIG_PATH=/usr/src/app/data/models/ssd_mobilenet_v2/ssd_mobilenet_v2_coco.config
export NUM_TRAIN_STEPS=40000
if [ -d /usr/src/app/data/train_data ]; then
    rm -r /usr/src/app/data/train_data
fi
if [ ! -d /usr/src/app/data/train_data ]; then
    mkdir /usr/src/app/data/train_data
fi

cd /usr/src/app/models/research
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr