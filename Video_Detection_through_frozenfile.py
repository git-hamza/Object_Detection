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
parser.add_argument('-i', dest='input_video_path',
                    help='video to be processed')
parser.add_argument('-o', dest='vid_record_path',
                    help='recorded video path')
args = parser.parse_args()

##Input and output video path
INPUT_VIDEO_PATH =  args.input_video_path
VID_RECORD_PATH = args.vid_record_path

print ('input_image_path     =', INPUT_VIDEO_PATH)
print ('saved_image_path     =', VID_RECORD_PATH )

##Threshold at which the detection bounding boxes will display
TH = 0.70

# Gloabl Variables
image_tensor = None
detection_boxes = None
detection_scores = None
detection_classes = None
num_detections = None

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt'

## number of classes that frozen file is trained for
NUM_CLASSES = 22

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

##funtion for initializing the video
def VideoSrcInit(paath):
    cap = cv2.VideoCapture(paath)
    flag, image = cap.read()
    if flag:
        print("Valid Video Path. Lets move to detection!")
    else:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")
    return cap

def VideoRecInit(WIDTH,HEIGHT,paath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(paath, fourcc, 3.0, (WIDTH,HEIGHT))
    return videowriter

##initialize the video reader and writer
cap = VideoSrcInit(args.input_video_path)
flag, image = cap.read()
# image = resize_img(image)
(ht,wd,_) = image.shape
videowriter = VideoRecInit(wd,ht,args.vid_record_path)

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
        frame_no = 0
        while True:
            frame_no += 1
            print ('frame_no: ' + str(frame_no))
            flag, image = cap.read()
            if flag == False:
                break
            timer = cv2.getTickCount()
            # image = resize_img(image)
            image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
            	[detection_boxes, detection_scores, detection_classes, num_detections],
            	feed_dict={image_tensor: image_np_expanded})

            sboxes = np.squeeze(boxes);
            sclasses = np.squeeze(classes).astype(np.int32);
            sscores = np.squeeze(scores);
            vis_util.visualize_boxes_and_labels_on_image_array(image,sboxes,sclasses, 
                sscores,s_category_index,min_score_thresh=TH,max_boxes_to_draw=100,use_normalized_coordinates=True,
                skip_scores=False,line_thickness=6)
            
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            print("fps : " + str(fps))
            # image_np=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # cv2.putText(image, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            videowriter.write(image)
        videowriter.release()
        cap.release()
        fid.close


