if [ d "mkdir /usr/src/app/data/train_data" ]; then
    rm -r /usr/src/app/data/train_data/*
fi
if [ -d "mkdir /usr/src/app/data/train_data" ]; then
    mkdir /usr/src/app/data/train_data
fi

cd /usr/src/app/models/research
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --train_dir=${MODEL_DIR} \