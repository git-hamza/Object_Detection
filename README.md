# Object_Detection
Folder structure for Object detection. This repo is tested on **tensorflow version 1.14** although will work with any tf1 version.
Please keep all the folder names as it is mentioned. e.g. annotation, images etc.

## Installation:
Install the Tensorflow Object Detection API dependencies from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md).
## Directories Before Training:
###### config directory : 
Configuration file needs to be copied inside this directory. Inside the configuration file you need to set the address for *train.record* and *test/val.record* and *label_map.pbtxt* files. You have to set the address for the **pretrained model checkpoint file**, **number of classes**, **batchsize**. You can also tweak other hyper parameter, If you have any knowledge about them.
###### data directory :
You data needs to be copied inside this directory i.e. annotations and images. This directory contain three *.py* files.
*xml_to_csv.py* :
>It converts all xml to a csv file

*generate_labelmap_from_csv.py*
>It will generate a label_map.pbtxt

*generate_tfrecord.py*
>It will generate a tfrecord file for the data that the model will take as an input

###### inputs directory : 
inputs directory should have two things before starting the training i.e. a *label* file and a *tfrecord* file.
###### object_detection directory:
We need to create softlink of object_detection from models directory.
Clone the models from the tensorflow github repository to your home directory through `git clone https://github.com/tensorflow/models`. From there we will make softlink of object_detection to our repository directory(where our directories and executable file lies). 
Use the following command to make softlink 
```
ln -s /home/hamza/models/research/object_detection /home/hamza/Object_Detection
```
It actually work as ` ln -s file_address address_where_to_link`.
###### Pre-trained_model: 
Training model for the first time, you need the checkpoints of a pretrained model that you are using. You can download it from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). What I did here is to make a directory in home by the name of *pre-trained_models* where I keep all my pre-trained model directories. From there I load their checkpoint into the configuration file. That way it is easy and more useful.
## Executable files:
###### run_train : 
it is an executable file that uses *train.py* to train the model and take configuration file as an input, please set the input of configuration file as *config/FILE_NAME.config*.
After executing, It will make a directory named as train where it saves all the checkpoints. It is designed in such a way through confiuration file, if training process is stopped in midway and few checkpoints are saved in the train directory. If you want to start the training again just run the *run_train* again and it will continue from the recent checkpoint.
###### run_eval : 
it is also an executable file that uses eval.py to evaluate the model from recent checkpoint. The recent checkpoint is loaded from train directory in it. How the evaluation will be performed are set at the end of the configuration file. It saves the output in a directory name eval.
###### run_export : 
it is also an executable file that uses export inference_graph.py and trained checkpoints from the train directory and output a directory named *output_inference_graph.pb* which has frozen graph of the model i.e. .pb file.
## Things to look for before training:
###### configuration file: 
Check the number of classes in the configuration file to make sure you are training for the amount that is written in this file. Check the *checkpoint address*, *train.record* and *label_map.pbtxt* addresses in training and evaluation part.
###### label_map.pbtxt : 
This file contain dictionary where labels are assigned to each class.
## Running process :
After setting the directory structure with the names mentioned in this repository, you can run the following commands step by step.
Note: if the *run_train*, *run_eval* or *run_export* is not running through those command, it means they are not executable yet. to make them exeuctable try this command `chmod +x FILE_NAME`.
```
1. python3 xml_to_csv.py
3. python3 generate_labelmap_from_csv.py
2. python3 generate_tfrecord.py --csv_input=train.csv  --output_path=../inputs/train.record --label_map=inputs/label_map.pbtxt
3. python3 generate_tfrecord.py --csv_input=val.csv  --output_path=../inputs/val.record --label_map=inputs/label_map.pbtxt
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
###### ImagesBatch_Detection_through_frozenfile.py : 
This is a python file that help you run inference with the help of frozen file. You just have to set the path of frozen file and label_map.pbtxt and set the threshold. You can run the file as
```python3 ImagesBatch_Detection_through_frozenfile.py -i Path_of_Input_Image_Folder -o Path_of_Output_Directory```.
###### Video_Detection_through_frozenfile.py : 
This is a python file that help you run inference with the help of frozen file. You just have to set the path of frozen file and label_map.pbtxt and set the threshold. You can run the file as
```python3 Video_Detection_through_frozenfile.py -i Path_of_Input_Video -o Path_of_Output_Video```.

###### Note :
Loading with multiple tfrecords file.
You can simply assign list of the file path by changing config file from train_input_reader: {
```
 tf_record_input_reader {
   input_path: "PATH_TO_BE_CONFIGURED/train.record"
 }
 label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
}
```
to 
```
train_input_reader: {
 tf_record_input_reader {
   input_path: ["PATH_TO_BE_CONFIGURED/train_a.record",
                "PATH_TO_BE_CONFIGURED/train_b.record"]
 }
 label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
}
```
this change may only work when multiple tfrecord files use the same label_map.pbtxt file.
