if [ -d "mkdir /usr/src/app/data/train_data" ]; then
    rm -rf /usr/src/app/data/train_data
fi
mkdir /usr/src/app/data/train_data
cd /usr/src/app/models/research
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr