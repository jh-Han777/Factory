avg_pre = np.zeros((1024,2048,3))
with open("/media/ssd1/cityscape/VOC2007/ImageSets/Main/train_s.txt",'r') as f:
    idx = f.readlines()
for item in idx:
    item = item.strip()+".jpg"
    img = cv2.imread(os.path.join("/media/ssd1/cityscape/VOC2007/JPEGImages",item),cv2.IMREAD_COLOR)
    avg_pre += img

avg = avg_pre / len(idx)
