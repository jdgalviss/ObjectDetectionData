"""
References:
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/exporting_models.md
https://github.com/tensorflow/models/issues/1988
Unfortunately, the tutorial for saving a model for inference "freezes" the
variables in place and makes them unservable by tensorflow_serving.
export_inference_graph.py exports an empty "variables" directory, which needs to
be populated.
The below script, which is a modified version of export_inference_graph, will
save the model in an "unfrozen" state, servable via TensorFlow Serving.
"""

"""Tool to export an object detection model for inference.
Prepares an object detection tensorflow graph for inference using model
configuration and an optional trained checkpoint. Outputs inference
graph, associated checkpoint files, a frozen inference graph and a
SavedModel (https://tensorflow.github.io/serving/serving_basic.html).
The inference graph contains one of three input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [None, None, None, 3]
  * `encoded_image_string_tensor`: Accepts a 1-D string tensor of shape [None]
    containing encoded PNG or JPEG images. Image resolutions are expected to be
    the same if more than 1 image is provided.
  * `tf_example`: Accepts a 1-D string tensor of shape [None] containing
    serialized TFExample protos. Image resolutions are expected to be the same
    if more than 1 image is provided.
and the following output nodes returned by the model.postprocess(..):
  * `num_detections`: Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`: Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores`: Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.
  * `detection_masks`: Outputs float32 tensors of the form
      [batch, num_boxes, mask_height, mask_width] containing predicted instance
      masks for each box if its present in the dictionary of postprocessed
      tensors returned by the model.
Notes:
 * This tool uses `use_moving_averages` from eval_config to decide which
   weights to freeze.
Example Usage:
--------------
data/TFRecords/Red
im = Image.open("/usr/src/app/data/TFRecords/Red/0.0_85.png")

predictor = tf.contrib.predictor.from_saved_model("/usr/src/app/data/capstone_models/mobilenetv1/saved_model/")

predictor = tf.contrib.predictor.from_saved_model("/models/mobilenetv1/mobilenetv1/saved_model/")
python freeze_model.py \
    --input_type image_tensor \
    --pipeline_config_path /usr/src/app/data/models/mobilenetv1/ssd_mobilenet_v1_coco.config  \
    --trained_checkpoint_prefix /usr/src/app/data/trained_models/mobilenetv1/train_data/model.ckpt-5000 \
    --output_directory /usr/src/app/data/capstone_models/mobilenetv1

python freeze_model.py \
    --input_type image_tensor \
    --pipeline_config_path /usr/src/app/data/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco.config  \
    --trained_checkpoint_prefix /usr/src/app/data/train_data/model.ckpt-200 \
    --output_directory /usr/src/app/data/capstone_models/inceptionv2

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - graph.pbtxt
 - model.ckpt.data-00000-of-00001
 - model.ckpt.info
 - model.ckpt.meta
 - frozen_inference_graph.pb
 + saved_model (a directory)
"""

import logging
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.exporter import (input_placeholder_fn_map,
                                       _add_output_tensor_nodes,
                                       _write_graph_and_checkpoint)


flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')

FLAGS = flags.FLAGS


def _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs):
    """Writes SavedModel to disk.
    Args:
      saved_model_path: Path to write SavedModel.
      trained_checkpoint_prefix: path to trained_checkpoint_prefix.
      inputs: The input image tensor to use for detection.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    saver = tf.train.Saver()
    with session.Session() as sess:
        saver.restore(sess, trained_checkpoint_prefix)
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

        tensor_info_inputs = {
              'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
        tensor_info_outputs = {}
        for k, v in outputs.items():
            tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

        detection_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                  inputs=tensor_info_inputs,
                  outputs=tensor_info_outputs,
                  method_name=signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.tag_constants.SERVING],
              signature_def_map={
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      detection_signature,
              },
          )
        builder.save()


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            optimize_graph=False,
                            output_collection_name='inference_op'):
    """Export helper."""
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    if input_type not in input_placeholder_fn_map:
        raise ValueError('Unknown input type: {}'.format(input_type))
    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type]()
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)
    outputs = _add_output_tensor_nodes(postprocessed_tensors,
                                       output_collection_name)

    saver = None
    if use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()

    _write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=trained_checkpoint_prefix)

    _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           optimize_graph=False,
                           output_collection_name='inference_op'):
    """Exports inference graph for the model specified in the pipeline config.
    Args:
      input_type: Type of input for the graph. Can be one of [`image_tensor`,
        `tf_example`].
      pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
      trained_checkpoint_prefix: Path to the trained checkpoint file.
      output_directory: Path to write outputs.
      optimize_graph: Whether to optimize graph using Grappler.
      output_collection_name: Name of collection to add output tensors to.
        If None, does not add output tensors to a collection.
    """
    detection_model = model_builder.build(pipeline_config.model,
                                          is_training=False)
    _export_inference_graph(input_type, detection_model,
                            pipeline_config.eval_config.use_moving_averages,
                            trained_checkpoint_prefix, output_directory,
                            optimize_graph, output_collection_name)


def main(_):
    assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing'
    assert FLAGS.trained_checkpoint_prefix, (
           '`trained_checkpoint_prefix` is missing')
    assert FLAGS.output_directory, '`output_directory` is missing'

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory)


if __name__ == '__main__':
    tf.app.run()