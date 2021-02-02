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

cap = cv2.VideoCapture("./Input/video.mp4")


class_names = [c.strip() for c in open("./class_names.txt").readlines()]

# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0

font = cv2.FONT_HERSHEY_SIMPLEX

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if ret == True:
        image, image_data = preprocess_image(frame, (416, 416))

        output_0, output_1 = tf_sess.run([tf_output_0,tf_output_1], feed_dict={
                tf_input: image_data[None, ...]})

        output_tensors = [output_0, output_1]

        output_image = myOwnPredict(image=image, output_tensors=output_tensors, \
                                    class_names=class_names,confidence=0.5, iou_threshold=0.4)
        # calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)

        cv2.putText(output_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Drone Detection', cv2.resize(output_image, (1200, 800)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break