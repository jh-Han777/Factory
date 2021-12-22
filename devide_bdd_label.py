import json

with open('/media/sda1/paper3/data/bdd100k/labels/bdd100k_labels_images_train.json','r') as f:
    labels_ = json.load(f)
##night
labels = [label['name']+'\n'for label in labels_ if label['attributes']['timeofday'] == 'night']

##daytime
#labels = [label['name']+'\n'for label in labels_ if label['attributes']['timeofday'] == 'daytime']

##dawn/dusk
#labels = [label['name']+'\n'for label in labels_ if label['attributes']['timeofday'] == 'dawn/dusk']

with open("/media/sda1/paper3/data/bdd100k/ImageSets/train_night.txt",'w') as f:
    f.writelines(labels)
