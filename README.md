# Object_Detection
Folder structure of Object detection
Please keep all the folder names as it is mentioned. e.g. annotation, images etc.

=> config directory : Configuration file needs to be copied here. Inside the configuration file you need to set the addresses for train.record and test/val.record and label_map.pbtxt file which contains the label of the classes. You also have to set the address for the pretrained model check point file.

=> data directory : data directory contain two .py files. xml_to_csv.py converts the annotations to csv file named as train.csv. generate_tfrecord file convert csv files to tf record file name as train.record and if you have test.csv or val.csv, you will have test/val.record too. You just need to copy the images and annotations folder to the data directory. For this repo we make val.csv by making a copy of train.csv and delete 80 percent of the image data which results in 20 percent of the training data in the test/val.csv. We use the name val.csv which produce val.record. If you change the name here, keep in mind you also have to edit the configuration file too.

=> inputs directory : inputs directory must have three things before starting the training. label_map.pbtxt which contain labels of the classes. train.record file and val.record file.

=> run_train : it is an executable file that uses train.py to train the model and take configuration file as an input please set the input of configuration file as config/FILE_NAME.config. it also make a directory named as train where it saves all the checkpoints. It is designed in such a way through confiuration file where if training process is stopped in midway and few checkpoints are saved in the train directory. If you start the training again, it will continue from the recent checkpoints.

=> run_eval : it is also an executable file that uses eval.py to evaluate the model from recent checkpoint. The recent checkpoint is loaded from train directory in it. How the evaluation will be performed are set at the end of the configuration file. It saves the output in a directory name eval.

=> run_export : it is also an executable file that uses export inference_graph.py and trained checkpoints from the train directory and output a directory named output_inference_graph.pb which has frozen graph of the model i.e. .pb file.

Softlink object_detection from models:
First clone the models from the tensorflow git to your home directory through git clone https://github.com/tensorflow/models. From there we will make softlink of object_detection to our repository directory. Use the following command to make softlink ln -s /home/hamza/models/research/object_detection /home/hamza/Object_Detection. it actually work as ln -s file_address address_where_to_link.
Running process :
After setting the directory structure with the names mentioned in this repository, you can run the following commands step by step.
Note: if the run_train, run_eval or run_export is not running through those command, it mean they are not executable yet. to make them exeuctable try this command chmod +x FILE_NAME

1. python3 xml_to_csv.py
2. python3 python generate_tfrecord.py --csv_input=train.csv  --output_path=../inputs/train.record
3. python generate_tfrecord.py --csv_input=val.csv  --output_path=../inputs/val.record
4. ./run_train
5. ./run_eval
6. ./run_export

