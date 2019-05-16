# Object Detection Training Pipeline
This pipeline is a implementation of [tensorflow/models](https://github.com/tensorflow/models)'s object detection pipeline. a Dockerfile is provided to facilitate the process.

One sample tasks is implemented:

* Detection of objects in a sidewalk: for this, a set of 4000 prelabeled images is available. These can be downloaded so that TFRecords files can be generated for training (starting with a pre trained model) and validation.

If TFRecords are already available, jump to step 4 (TFRecords must be in the ./data/TFRecords folder - .config file in /data/models/xxxxxx/xxxxx.config must be configured - labels must be modified in /data/config folder)

## Requirements

1. docker
2. docker-compose
3. nvidia-docker

# Steps

1. After cloning the repository, build the container:

    ```shell
    > cd ObjectDetectionData
    > docker-compose build
    ```

2. Download images
    ```shell
    > docker-compose run --service-ports --rm download-images
    ```

3. Create tf records
    ```shell
    > docker-compose run --service-ports --rm create-tf-records
    ```

4. Train
    ```shell
    > docker-compose run --service-ports --rm train
    ```   

    The model used in this case is ssd_mobilenet_v2_coco.config. Pre-trained models can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    and .config files to configure the model in: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs, which will have to be configured according to the training hyperparameters, checkpoint locations, TFRecords locations, etc.

    These files must be saved inside a folder inside the ./data/models folder.

5. To access tensorboard, while the training docker container is running:

    * Get the docker container ID:

    ```shell
    > docker-ps
    ```   

    * Execute the a bash terminal in a docker-container:
    ```shell
    > docker exec -it (CONTAINER ID) bash
    ```   

    * Execute Tensorboard:
    ```shell
    > tensorboard --logdir=$MODEL_DIR
    ``` 

    * Acces tensorboard from browser under localhost:6006