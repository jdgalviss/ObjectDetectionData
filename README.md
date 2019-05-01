# Object Detection Training Pipeline

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