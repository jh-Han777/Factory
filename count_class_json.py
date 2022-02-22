import os
import json

with open("/media/hdd1/paper3/data/bdd100k/labels/bdd100k_labels_images_val.json") as f:
    annots = json.load(f)
##bdd100k
person = 0
rider = 0
car = 0
truck = 0
bus = 0
train = 0
motor = 0
bike = 0
light = 0
sign = 0

num_img = 0
for annot in annots:

    if annot['attributes']["timeofday"] == "daytime":
        num_img += 1
        print(num_img)
        labels = annot['labels']

        for label in labels:

            if label['category'] in ['drivable area', 'lane']:
                continue

            if label['category'] == 'car':
                car += 1
            elif label['category'] == 'rider':
                rider += 1
            elif label['category'] == 'person':
                person += 1
            elif label['category'] == 'truck':
                truck += 1
            elif label['category'] == 'bus':
                bus += 1
            elif label['category'] == 'train':
                train += 1
            elif label['category'] == 'motor':
                motor += 1
            elif label['category'] == 'bike':
                bike += 1
            elif label['category'] == 'traffic light':
                light += 1
            elif label['category'] == "traffic sign":
                sign += 1
            else:
                raise Exception("No class")

print("person:",person,"\nrider ",rider,"\ncar ",car, "\ntruck ",truck, "\nbus ", bus, "\ntrain ", train, "\nmotor ", motor, "\nbike ", bike,"\nlight",light,"\nsign",sign)
