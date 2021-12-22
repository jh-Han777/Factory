import os
import shutil

with open("/media/ssd1/cityscape/VOC2007/ImageSets/Main/test_s.txt",'r') as f:
    item = f.readlines()

for idx in item:
    idx = idx.strip()
    src = os.path.join("/media/ssd1/cityscape/VOC2007/JPEGImages/",idx+".jpg")
    dst = os.path.join("/media/ssd1/cityscape/VOC2007/source_test/",idx + ".jpg")
    shutil.copyfile(src,dst)
