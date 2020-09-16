##importing the neccesary packages
import argparse
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
import tensorflow as tf
print('Using Tensorflow Version ' + str(tf.__version__))
import cv2
import time
sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

##Argumentparser is used to run through the terminal 
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_image_batch',
                    help='image folder to be precessed')
parser.add_argument('-o', dest='output_directory',
                    help='directory path for output images')
args = parser.parse_args()

##Input and output video path
INPUT_DIR_PATH =  args.input_image_batch
Output_DIR_PATH = args.output_directory

if not os.path.isdir(Output_DIR_PATH):
    os.mkdir(Output_DIR_PATH)

##Threshold at which the detection bounding boxes will display
TH = 0.70

# Gloabl Variables
image_tensor = None
detection_boxes = None
detection_scores = None
detection_classes = None
num_detections = None

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'output_inference_graph.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'inputs/label_map.pbtxt'

## number of classes that frozen file is trained for
NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading hole label map
s_label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
s_categories = label_map_util.convert_label_map_to_categories(s_label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
s_category_index = label_map_util.create_category_index(s_categories)

def resize_img(frame):
    percent = 50
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    ##Loading image
    count = 0
    mypath = INPUT_DIR_PATH
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    for n in range(0,len(onlyfiles)):
        count +=1
        print("count : " + str(count))
        image = cv2.imread(join(mypath, onlyfiles[n]))
        # image = resize_img(image)
        image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        fetch_time = time.time()

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
        	[detection_boxes, detection_scores, detection_classes, num_detections],
        	feed_dict={image_tensor: image_np_expanded})

        sboxes = np.squeeze(boxes);
        sclasses = np.squeeze(classes).astype(np.int32);
        sscores = np.squeeze(scores);
        _= vis_util.visualize_boxes_and_labels_on_image_array(image,sboxes,sclasses,
          sscores,s_category_index,min_score_thresh=TH,
          max_boxes_to_draw=100,use_normalized_coordinates=True,
          skip_scores=False,line_thickness=6)
        print("Inference_Time : " + str(time.time()-fetch_time))
        cv2.imwrite(Output_DIR_PATH+onlyfiles[n],image)

