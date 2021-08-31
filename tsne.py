
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

def file_load(text_file):
    file_name = []

    with open(text_file,"r") as f:
        while True:
            line =f.readline()
            if not line:
                break
            file_name.append(line.strip())

    return file_name


class data_loader(Dataset):
    def __init__(self,data_dir,data_list):
        super(data_loader,self).__init__()
        self.data_dir = data_dir
        self.data_list = file_load(data_list)

    def __len__(self):
        return self.n_data

    def __getitem__(self,idx):
        img = cv2.imread(self.data_dir+self.data_list[idx])
        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
        return np.transpose(img,(2,1,0))


data_s = data_loader('/media/sda1/note/imageset/cityscape/','/media/sda1/note/data_s.txt')
data_t = data_loader('/media/sda1/note/imageset/foggy cityscape/','/media/sda1/note/data_t.txt')
data_mix = data_loader('/media/sda1/note/imageset/mixup/','/media/sda1/note/mix.txt')


def visualize(data_s,data_t,data_mix):
    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(data_s):
        img = test_data.squeeze()
        img = np.transpose(img, (1,2,0))
        source_label_list.append('0')
        source_img_list.append(img)

    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(data_t):
        img = test_data.squeeze()
        img = np.transpose(img, (1,2,0))
        target_label_list.append('1')
        target_img_list.append(img)

    mix_label_list = []
    mix_img_list = []
    for i, data_mix in enumerate(data_mix):
        img= data_mix.squeeze()
        img = np.transpose(img, (1,2,0))
        mix_label_list.append('2')
        mix_img_list.append(img)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_label_list.extend((mix_label_list))

    combined_img_list = source_img_list + target_img_list + mix_img_list

    source_domain_list = []
    target_domain_list = []
    mixup_domain_list = []
    for i in range(70):
        mixup_domain_list.append(2)
        target_domain_list.append(1)
        source_domain_list.append(0)
    combined_domain_list = mixup_domain_list + target_domain_list + source_domain_list
    print("Extract features to draw T-SNE plot...")

    tsne = TSNE(perplexity=200, n_components=2, init='pca', n_iter=1000)
    combined_img_list = np.array(combined_img_list,dtype= np.float64)
    combined_img_list=np.reshape(combined_img_list,(-1,28*28*3))
    dann_tsne = tsne.fit_transform(combined_img_list)

    print('Draw plot ...')
    plot_embedding(dann_tsne, combined_label_list, combined_domain_list)
    #plot_embedding(dann_tsne, combined_label_list)

def plot_embedding(X, y, d):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        elif d[i]==1:
            colors = (1.0, 0.0, 0.0, 1.0)
        elif d[i] ==2:
            colors = (0, 1.0, 0.0, 1.0)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.title('T-SNE')
    plt.show()

visualize(data_s,data_t,data_mix)