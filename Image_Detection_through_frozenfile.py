##importing the neccesary packages
import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print('Using Tensorflow Version ' + str(tf.__version__))
import zipfile
import re
import glob
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
import imutils

sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

##Argumentparser is used to run through the terminal 
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_image_path',
                    help='image to be processed')
parser.add_argument('-o', dest='saved_image_path',
                    help='saved image path')
args = parser.parse_args()

##Input and output video path
INPUT_IMG_PATH =  args.input_image_path
Saved_IMG_PATH = args.saved_image_path

print ('input_image_path     =', INPUT_IMG_PATH)
print ('saved_image_path     =', Saved_IMG_PATH)

##Threshold at which the detection bounding boxes will display
TH = 0.70

# Gloabl Variables
person_image_tensor = None
person_detection_boxes = None
person_detection_scores = None
person_detection_classes = None
person_num_detections = None

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PERSON_PATH_TO_CKPT = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
S_PATH_TO_LABELS = 'label_map.pbtxt'

## number of classes that frozen file is trained for
NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.
person_detection_graph = tf.Graph()
with person_detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PERSON_PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



# ## Loading hole label map
s_label_map = label_map_util.load_labelmap(S_PATH_TO_LABELS)
s_categories = label_map_util.convert_label_map_to_categories(s_label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
s_category_index = label_map_util.create_category_index(s_categories)



with person_detection_graph.as_default():
  with tf.Session(graph=person_detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    person_image_tensor = person_detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    person_detection_boxes = person_detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    person_detection_scores = person_detection_graph.get_tensor_by_name('detection_scores:0')
    person_detection_classes = person_detection_graph.get_tensor_by_name('detection_classes:0')
    person_num_detections = person_detection_graph.get_tensor_by_name('num_detections:0')
    
    ##Loading image
    image = cv2.imread(INPUT_IMG_PATH)
    print(image.shape)
    scale_percent = 40 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_np = cv2.cvtColor(cv2.resize(image,dim),cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    fetch_time = time.time()

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
    	[person_detection_boxes, person_detection_scores, person_detection_classes, person_num_detections],
    	feed_dict={person_image_tensor: image_np_expanded})

    sboxes = np.squeeze(boxes);
    sclasses = np.squeeze(classes).astype(np.int32);
    sscores = np.squeeze(scores);
    _,box= vis_util.visualize_boxes_and_labels_on_image_array(image_np,sboxes,sclasses,
      sscores,s_category_index,min_score_thresh=TH,
      max_boxes_to_draw=10,use_normalized_coordinates=True,
      skip_scores=False,line_thickness=2)
    image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    cv2.imwrite(Saved_IMG_PATH,image_np)

