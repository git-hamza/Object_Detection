import xml.etree.ElementTree as ET 
import os
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-c","--csv", type =str,help="path csv file")
args = vars(ap.parse_args())

Labels = {}
num = 1
fields = ['filename', 'width','height','class','xmin', 'ymin','xmax', 'ymax']
df = pd.read_csv(args["csv"], usecols=fields)

for i in range(0, len(df['class'])):
	txt = df['class'].iloc[i]
	if len(Labels) ==0:
		Labels[txt] = num
	else:
		if txt in Labels.keys():
			pass
		else:
			num=num+1
			Labels[txt] = num

print("Labels: ",Labels)
print("Number of Labels: ",len(Labels))

with open("../inputs/label_map.pbtxt","w") as file:           ##Generate a label_map in .pbtxt format which is used in inference
    for key,value in Labels.items():
        file.write("item {\n\tid : " + str(value) + "\n\tname : '" + str(key) + "'\n}\n")