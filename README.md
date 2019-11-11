﻿# Object_Detection
Folder structure of Object detection
Please keep all the folder names as it is mentioned. e.g. annotation, images etc.

## Installation:
First you have to install the Tensorflow Object Detection API dependencies from the following link https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md.
## Directories Before Training:
###### config directory : 
Configuration file needs to be copied here. Inside the configuration file you need to set the addresses for *train.record* and *test/val.record* and *label_map.pbtxt* file which contains the label of the classes. You also have to set the address for the pretrained model check point file.
###### data directory : 
data directory contain two .py files. *xml_to_csv.py* converts the annotations to csv file named as *train.csv*. *generate_tfrecord.py* file convert csv files to tf record file name as *train.record* and if you have *test.csv* or *val.csv*, you will have *test/val.record* too. You just need to copy the images and annotations folder to the data directory. For this repo we make *val.csv* by making a copy of *train.csv* and delete 80 percent of the image data which results in 20 percent of the training data in the *test/val.csv*. We use the name *val.csv* which produce *val.record*. If you change the name here, keep in mind you also have to edit the configuration file too.
###### inputs directory : 
inputs directory must have three things before starting the training. *label_map.pbtxt* which contain labels of the classes. *train.record* file and *val.record* file.
## Directories Before Training:
###### config directory : 
Configuration file needs to be copied here. Inside the configuration file you need to set the addresses for *train.record* and *test/val.record* and *label_map.pbtxt* file which contains the label of the classes. You also have to set the address for the pretrained model check point file.
###### data directory : 
data directory contain two .py files. *xml_to_csv.py* converts the annotations to csv file named as *train.csv*. *generate_tfrecord.py* file convert csv files to tf record file name as *train.record* and if you have *test.csv* or *val.csv*, you will have *test/val.record* too. You just need to copy the images and annotations folder to the data directory. For this repo we make *val.csv* by making a copy of *train.csv* and delete 80 percent of the image data which results in 20 percent of the training data in the *test/val.csv*. We use the name *val.csv* which produce *val.record*. If you change the name here, keep in mind you also have to edit the configuration file too.
###### inputs directory : 
inputs directory must have three things before starting the training. *label_map.pbtxt* which contain labels of the classes. *train.record* file and *val.record* file.
###### Softlink object_detection directory from models directory:
First clone the models from the tensorflow git to your home directory through git clone https://github.com/tensorflow/models. From there we will make softlink of object_detection to our repository directory(where our directories and executable file lies). Use the following command to make softlink 
`ln -s /home/hamza/models/research/object_detection /home/hamza/Object_Detection`
It actually work as ` ln -s file_address address_where_to_link`.
###### Pre-trained_model: 
Training model for the first time with pretrained weights, you need the checkpoints of a pretrained model that you are using. You can download it from here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md . What I do is to make a directory in home by the name of pre-trained model where I keep all my pre-trained model directories. From there I load their checkpoint into the configuration file. That way it is easy and more useful.
## Executable files:
###### run_train : 
it is an executable file that uses train.py to train the model and take configuration file as an input please set the input of configuration file as *config/FILE_NAME.config*. it also make a directory named as train where it saves all the checkpoints. It is designed in such a way through confiuration file where if training process is stopped in midway and few checkpoints are saved in the train directory. If you start the training again, it will continue from the recent checkpoints.
###### run_eval : 
it is also an executable file that uses eval.py to evaluate the model from recent checkpoint. The recent checkpoint is loaded from train directory in it. How the evaluation will be performed are set at the end of the configuration file. It saves the output in a directory name eval.
###### run_export : 
it is also an executable file that uses export inference_graph.py and trained checkpoints from the train directory and output a directory named *output_inference_graph.pb* which has frozen graph of the model i.e. .pb file.
## Things to look for before training:
###### configuration file: 
Check the number of classes in the configuration file to make sure you are training for the amount that is written in this file. Check the checkpoint address, *train.record* and *label_map.pbtxt* addresses in training and 
evaluation part.
###### generate_tfrecord.py: 
Please check *class_text_to_int(row_labels)* function in the file. It is the first function in the file and it assign the labels to the classes. Please make sure you have correctly label for every class that you have given in *label_map.pbtxt* file.
###### label_map.pbtxt : 
This file contain dictionary where labels are assigned to each class. Please check the label with those you provided in *generate_tfrecord.py*.
###### Pre-trained_model: 
Training model for the first time with pretrained weights, you need the checkpoints of a pretrained model that you are using. You can download it from here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md . What I do is to make a directory in home by the name of pre-trained model where I keep all my pre-trained model directories. From there I load their checkpoint into the configuration file. That way it is easy and more useful.
## Executable files:
###### run_train : 
it is an executable file that uses train.py to train the model and take configuration file as an input please set the input of configuration file as *config/FILE_NAME.config*. it also make a directory named as train where it saves all the checkpoints. It is designed in such a way through confiuration file where if training process is stopped in midway and few checkpoints are saved in the train directory. If you start the training again, it will continue from the recent checkpoints.
###### run_eval : 
it is also an executable file that uses eval.py to evaluate the model from recent checkpoint. The recent checkpoint is loaded from train directory in it. How the evaluation will be performed are set at the end of the configuration file. It saves the output in a directory name eval.
###### run_export : 
it is also an executable file that uses export inference_graph.py and trained checkpoints from the train directory and output a directory named *output_inference_graph.pb* which has frozen graph of the model i.e. .pb file.
## Things to look for before training:
###### configuration file: 
Check the number of classes in the configuration file to make sure you are training for the amount that is written in this file. Check the checkpoint address, *train.record* and *label_map.pbtxt* addresses in training and 
evaluation part.
###### generate_tfrecord.py: 
Please check *class_text_to_int(row_labels)* function in the file. It is the first function in the file and it assign the labels to the classes. Please make sure you have correctly label for every class that you have given in *label_map.pbtxt* file.
###### label_map.pbtxt : 
This file contain dictionary where labels are assigned to each class. Please check the label with those you provided in *generate_tfrecord.py*.
## Running process :
After setting the directory structure with the names mentioned in this repository, you can run the following commands step by step.
Note: if the *run_train*, *run_eval* or *run_export* is not running through those command, it mean they are not executable yet. to make them exeuctable try this command `chmod +x FILE_NAME`.
```
1. python3 xml_to_csv.py
2. python3 generate_tfrecord.py --csv_input=train.csv  --output_path=../inputs/train.record
3. python3 generate_tfrecord.py --csv_input=val.csv  --output_path=../inputs/val.record
4. ./run_train
5. ./run_eval
6. ./run_export
```
## Directories After Training:
###### train: 
After you run the executable *./run_train* successfully. A directory by the name of train will be created in you main directory that will contain all you saved checkpoints.
###### eval : 
After training, you can run the executable *./run_eval* which will create a directory by the name of eval and it will have the evaluted images from the *test/val*.
###### output_inference_graph.pb : 
After you training when you are reading to export the freezing graph. You run the *./run_export* which will create the *output_inference_graph.pb* directory. Inside this directory you will have your frozen file with .pb extension.
## Inference test of the frozen graph:
###### Image_Detection_through_frozenfile.py : 
This is a python file that help you run inference with the help of frozen file. You just have to set the path of frozen file and label_map.pbtxt and set the threshold. You can run the file as
```python3 Image_Detection_through_frozenfile.py -i Path_of_Input_Image -o Path_of_Output_Image```.
###### Video_Detection_through_frozenfile.py : 
This is a python file that help you run inference with the help of frozen file. You just have to set the path of frozen file and label_map.pbtxt and set the threshold. You can run the file as
```python3 Video_Detection_through_frozenfile.py -i Path_of_Input_Video -o Path_of_Output_Video```.
