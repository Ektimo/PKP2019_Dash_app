import tensorflow as tf
import tensorflow.contrib.resampler
import numpy as np
from PIL import Image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# load input image
# imagePath = "../test_images/car.jpg"
imagePath = "../test_images/guy1.jpg"

inputImage = Image.open(imagePath)
w, h = inputImage.size
inputImage = inputImage.resize((320, 320), Image.ANTIALIAS)
# inputImage = inputImage.crop((159, 159, w-160, h-160))
# inputImage.show()
# print(inputImage.size)
imageNp = load_image_into_numpy_array(inputImage)

# normalizeImage = True
# if normalizeImage:
#     imageNp = (imageNp * (2.0 / 255.0) - 1.0).astype(np.uint8)

imageNpExpanded = np.expand_dims(imageNp, axis=0)
print("model input image shape: ", imageNpExpanded.shape)


modelDir = "/home/student/Documents/AnonAI/face_detection/models/mobilenet_models/"
modelFile = modelDir + "mobilenet_ssd_v2_face_quant_postprocess.tflite"

interpreter = tf.contrib.lite.Interpreter(model_path=modelFile)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()  # list with len 4

print("input required shape: ", input_details[0]['shape'])

interpreter.set_tensor(input_details[0]['index'], imageNpExpanded)
interpreter.invoke()

print("input details: ")
for key, value in input_details[0].items() :
    print (key, value)
print("\n\n")

outputData_0 = interpreter.get_tensor(output_details[0]['index'])
outputData_1 = interpreter.get_tensor(output_details[1]['index'])
outputData_2 = interpreter.get_tensor(output_details[2]['index'])
outputData_3 = interpreter.get_tensor(output_details[3]['index'])

print("output data: ", outputData_0)
# print("output data: ", outputData_1)
# print("output data: ", outputData_2)
# print("output data: ", outputData_3)

for c_output_details in output_details:
    print(c_output_details['name'])

print("\n\n")
for key, value in output_details[0].items() :
    print (key, value)
print("\n\n")
for key, value in output_details[1].items() :
    print (key, value)
print("\n\n")
for key, value in output_details[2].items() :
    print (key, value)
print("\n\n")
for key, value in output_details[3].items() :
    print (key, value)

res = np.squeeze(outputData_0)
print(res)

# display details of all tensors in the model
tensorDetails = interpreter.get_tensor_details()

for tensor in tensorDetails:
    print("\n\n")
    for key, det in tensor.items():
        print(key, ": ", det)

print("\n\n")
print("concat tensor values:")
print(interpreter.get_tensor(302))

# for idx in range(0, 400):
#     t = interpreter.get_tensor(idx)
#     print(idx)
#     print(type(t))
