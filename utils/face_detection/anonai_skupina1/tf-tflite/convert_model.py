import tensorflow as tf
from tensorflow.python.platform import gfile

dir = "./facessd_mobilenet_v2_quantized_320x320_open_image_v4/"
# dir = "./mobilenet/"

# this script converts the input tensorflow model into an .tflite file
inputGraphDef = dir + "tflite_graph.pb"
inputGraphDefPbtxt = dir + "tflite_graph.pbtxt"
outTfLite = dir + "tlfite_output.tflite"


# results in key error: ParallelInterleaveDataset
ckptMeta = dir + "model.ckpt.meta"
# ckptMeta = dir + "mobilenet_v1_1.0_224_quant.ckpt.meta"

ckptData = dir + "model.ckpt.data-00000-of-00001"
# ckptData = dir + "mobilenet_v1_1.0_224_quant.ckpt.data-00000-of-00001"

# tf.contrib.resampler
# with tf.Session() as sess:
#   saver = tf.train.import_meta_graph(ckptMeta, clear_devices=True)
#   saver.restore(sess, ckptData)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckptMeta, clear_devices=True)
    saver.restore(sess, dir + "model.ckpt")
    # saver.restore(sess, dir + "mobilenet_v1_1.0_224_quant.ckpt")

    # convert to tflite

    # converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors ?????, [outTfLite])
    tflite_model = converter.convert()

# input_shapes must be defined for this model
# i dont know what values are supposed to be in input/output_arrays
# converter = tf.lite.TFLiteConverter.from_frozen_graph(inputGraphDef, ["model_inputs"], ["model_outputs"], input_shapes=(320,320))

# saved model does not exist
# converter = tf.lite.TFLiteConverter.from_saved_model(inputGraphDefPbtxt)

# fails cause: NotFoundError: Op type not registered 'TFLite_Detection_PostProcess'
# with tf.Session() as sess:
#   with gfile.FastGFile(inputGraphDef,'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')


