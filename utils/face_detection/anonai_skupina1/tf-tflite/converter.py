import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #print(print([i for i in dir(graph_def) if i[0] != '_']))
    with tf.Graph().as_default() as graph:
        import tensorflow.contrib.resampler
        tf.import_graph_def(graph_def, name="prefix")
    

# load_graph('facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb')

"""
nodes = [op.name for op in graph_def.get_operations()]
print(nodes)
"""

"""

sess = tf.Session()
path = '/home/student/Documents/AnonAI/face_detection/code/tf_to_tflite/facessd_mobilenet_v2_quantized_320x320_open_image_v4'
saver = tf.train.import_meta_graph(path + '/model.ckpt.meta')
saver.restore(sess,path + "/model.ckpt.data-00000-of-00001")
"""