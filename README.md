# Object Detection Data Pipeline

# Steps

1. Create CSV, images,frozen_model, output, TFRecords, train directories using the following command in the project root:

```shell
> bash initializeProject
```
2. Run `script/get_images.py` notebook to get dataset images

3. Run `script/json_to_csv.py` notebook to convert the annotation files into a train and validation CSV files

4. Run `script/csv_to_tf.py` notebook to convert train and validation CSV into a tensorflow record.


TF Records and labels file have been created.



