import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('./model/trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

for node in trt_graph.node:
  if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

input_names = ['input_1']
output_names = ['conv2d_10/BiasAdd', 'conv2d_13/BiasAdd']

input_tensor_name = input_names[0] + ":0"
output_0_tensor_name = output_names[0] + ":0"
output_1_tensor_name = output_names[1] + ":0"

tf_input = tf_sess.graph.get_tensor_by_name(input_tensor_name)
tf_output_0=tf_sess.graph.get_tensor_by_name(output_0_tensor_name) #tensor 13x13x18
tf_output_1=tf_sess.graph.get_tensor_by_name(output_1_tensor_name) #tensor 26x26x18
#===================================================================================
from predict import myOwnPredict, preprocess_image
import cv2
import numpy as np

IMAGE_PATH = "./test_drone.jpg"

orig = cv2.imread(IMAGE_PATH)
image, image_data = preprocess_image(orig, (416, 416))

class_names = [c.strip() for c in open("./class_names.txt").readlines()]

output_0, output_1 = tf_sess.run([tf_output_0,tf_output_1], feed_dict={
        tf_input: image_data[None, ...]})

output_tensors = [output_0, output_1]
print(output_0.shape)
print(output_1.shape)
myOwnPredict(image=image, output_tensors=output_tensors, class_names=class_names,confidence=0.5, iou_threshold=0.4)