import json
import os
import csv
import random
import math

files = !ls ../annotations
path = '../annotations/'
savePath = '../CSV/'
width = "640"
height = "480"

images = []
print("Parsing json to csv...")
for fname in files:
    
    with open(path + fname) as json_file:
        
        data = json.load(json_file)
        
        for image in data:
            images.append(image)
            


random.shuffle(images)

trainLen = math.ceil(len(images)*0.8)

train = images[0:trainLen]
val = images[trainLen:]

#Train
createPath = savePath + "train.csv"

with open(createPath, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
    filewriter.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    
    for image in train:
        
        url = image['url']
        name = url.split('/')[-1]


        if image['labels'] != None:

            for label in image['labels']:

                category = label['category']
                xmin = label['box2d']['x1']
                ymin = label['box2d']['y1']
                xmax = label['box2d']['x2']
                ymax = label['box2d']['y2']

                l = [name, width, height, category, xmin, ymin, xmax, ymax]
                filewriter.writerow(l)
    
    
#Validation
createPath = savePath + "validation.csv"

with open(createPath, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
    filewriter.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    
    for image in val:
        
        url = image['url']
        name = url.split('/')[-1]

        if image['labels'] != None:

            for label in image['labels']:

                category = label['category']
                xmin = label['box2d']['x1']
                ymin = label['box2d']['y1']
                xmax = label['box2d']['x2']
                ymax = label['box2d']['y2']

                l = [name, width, height, category, xmin, ymin, xmax, ymax]
                filewriter.writerow(l)
print("csv created")