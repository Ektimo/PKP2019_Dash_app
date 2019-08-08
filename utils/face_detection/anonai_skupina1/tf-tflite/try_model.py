import tensorflow as tf
import tensorflow.contrib.resampler
import numpy as np

modelDir = "/home/student/Documents/AnonAI/face_detection/models/mobilenet_models/"
modelFile = modelDir + "mobilenet_ssd_v2_face_quant_postprocess.tflite"

"""
with tf.Session() as sess:
    
    graphDef = tf.GraphDef()
    with tf.gfile.GFile(modelFile, "rb") as f:
        graphDef.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graphDef, name="prefix")
"""

interpreter = tf.contrib.lite.Interpreter(model_path=modelFile)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print("\n", output_details, "\n")

# Test model on random input data.
input_shape = input_details[0]['shape']
print("input data shape: ", input_shape)

# change the following line to feed into your own data.
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
# print(input_data[0])

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
