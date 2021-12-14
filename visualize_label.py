import cv2
import xml.etree.ElementTree as ET
import os

img1 = cv2.imread("/media/hdd1/note/imageset/cityscape/source_bremen_000154_000019_leftImg8bit.jpg")
img1_path = "source_bremen_000154_000019_leftImg8bit"

tree = ET.parse(os.path.join('/media/hdd1/iitp/CR-DA-DET/SW_Faster_ICR_CCR/data/cityscape/VOC2007/Annotations',img1_path+'.xml'))
root = tree.getroot()

for obj in root.iter("object"):
    xmin = int(obj.find("bndbox").findtext("xmin"))
    xmax = int(obj.find("bndbox").findtext("xmax"))
    ymin = int(obj.find("bndbox").findtext("ymin"))
    ymax = int(obj.find("bndbox").findtext("ymax"))
    class_name = obj.findtext("name")
    area = int((xmax - xmin) * (ymax - ymin))
    cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.putText(
            img1,
            "%s %d" % (class_name, area),
            (xmin, ymin),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (255, 0, 0),
            thickness=2
        )

cv2.imshow("",img1)
cv2.waitKey(0)
