import xml.etree.ElementTree as ET
import os


person = 0
rider = 0
car = 0
truck = 0
bus = 0
train = 0
motorcycle = 0
bicycle = 0

with open("/media/hdd1/paper3/data/cityscape/VOC2007/ImageSets/Main/train_s.txt") as f:
    sets = f.readlines()

for idx,item in enumerate(sets):
    filename = os.path.join("/media/hdd1/paper3/data/cityscape/VOC2007/Annotations",item.strip() + ".xml")
    print(idx)
    tree = ET.parse(filename)
    objs = tree.findall("object")
    ## cityscape
    for obj in objs:
        _class = obj.find("name").text
        if _class == "person":
            person +=1
        elif _class == "rider":
            rider += 1
        elif _class == "car":
            car += 1
        elif _class == "truck":
            truck += 1
        elif _class == "bus":
            bus += 1
        elif _class == "train":
            train += 1
        elif _class == "motorcycle":
            motorcycle += 1
        elif _class == "bicycle":
            bicycle += 1
        else:
            raise Exception("not class")
print("person:",person,"\nrider ",rider,"\ncar ",car, "\ntruck ",truck, "\nbus ", bus, "\ntrain ", train, "\nmotorcycle ", motorcycle, "\nbicycle ", bicycle)
