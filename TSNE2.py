from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pickle
import pprint
import sys
import time

import _init_paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.da_faster_rcnn.resnet import resnet
#from model.da_faster_rcnn.vgg16 import vgg16

#switch model
from gan.vgg_gan15 import vgg16
#from last.vgg16_last_align import vgg16
# from model.faster_rcnn.vgg16 import vgg16
#from model.da_faster_rcnn.resnet import resnet

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import load_net, save_net, vis_detections
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch.autograd import Variable

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print(sys.path)


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="cityscape",
        type=str,
    )
    parser.add_argument(
        "--num_epoch", dest="num_epoch", help="save res", default=-1, type=int,
    )
    parser.add_argument(
        "--output_dir", dest="output_dir", help="resoutput", default="./", type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default="cfgs/vgg16.yml",
        type=str,
    )
    parser.add_argument(
        "--net",
        dest="net",
        help="vgg16, res50, res101, res152",
        default="vgg16",
        type=str,
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # parser.add_argument('--load_dir', dest='load_dir',
    #                     help='directory to load models', default="models",
    #                     type=str)
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        help="directory to load models",
        default="models.pth",
        type=str,
    )
    parser.add_argument(
        "--part",
        dest="part",
        help="test_s or test_t or test_all",
        default="test_t",
        type=str,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true"
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )
    parser.add_argument(
        "--parallel_type",
        dest="parallel_type",
        help="which part of model to parallel, 0: all, 1: model before roi pooling",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--checksession",
        dest="checksession",
        help="checksession to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkepoch",
        dest="checkepoch",
        help="checkepoch to load network",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load network",
        default=10021,
        type=int,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="model file name",
        default="res101.bs1.pth",
        type=str,
    )
    parser.add_argument(
        "--vis", dest="vis", help="visualization mode", action="store_true"
    )

    parser.add_argument(
        "--USE_cls_cotrain",
        dest="USE_cls_cotrain",
        help="USE_cls_cotrain",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--USE_box_cotrain",
        dest="USE_box_cotrain",
        help="USE_box_cotrain",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--lc",
        dest="lc",
        help="whether use context vector for pixel level",
        action="store_true",
    )
    parser.add_argument(
        "--gc",
        dest="gc",
        help="whether use context vector for global level",
        action="store_true",
    )
    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--resume_name",
        dest="resume_name",
        help="resume checkpoint path",
        default="",
        type=str,
    )


    #--------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    net = "vgg16"
    part = "test_t"
    output_dir = "/mnt/sdb1/jh_han/etri/tsne/"
    dataset = "mskda_bdd"
    model_dir = "/mnt/sdb1/jh_han/etri/save_model/train_gan15/mskda_bdd_10.pth"
    epoch = 20

    command = "--cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, epoch)
    # --------------------------------------------
    args = parser.parse_args(command.split())


    """
    args = parser.parse_args()
    """
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def flat(data):
    #data = torch.tensor(data)
    data = data.squeeze()
    shape = data.shape
    data = data.view(shape[0], -1)
    return data.cpu()

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range

color=["blue", "green", "red"]

def visualize_mul(data):
    for domain in range(3):
        x = data[domain, :, 0]
        y = data[domain, :, 1]
        plt.scatter(x, y, c=color[domain])
    plt.show()

if __name__ == "__main__":
    project_path = "/mnt/sdb1/jh_han/etri/"
    args = parse_args()

    print("Called with args:")
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[4,8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]

    elif args.dataset == "cityscape":
        print("loading our dataset...........")
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscape_2007_test_t"
        args.all_imdbtest_name = "cityscape_2007_test_all"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "mskda_bdd":
        print("loading our dataset...........")
        args.s1_imdb_name = "bdd100k_daytime_train"
        args.s2_imdb_name = "bdd100k_night_train"
        args.s1_imdbtest_name = "bdd100k_daytime_val"
        args.s2_imdbtest_name = "bdd100k_night_val"
        args.t_imdb_name = "bdd100k_dd_train"
        args.t_imdbtest_name = "bdd100k_dd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "water":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "water_train"
        args.t_imdbtest_name = "water_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "clipart":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_trainval"
        args.t_imdbtest_name = "clipart_trainval"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "comic":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "comic_train"
        args.t_imdbtest_name = "comic_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "bdd":
        print("loading our dataset...........")
        args.s_imdb_name = "citybdd7_train"
        args.t_imdb_name = "bdd_train"
        # args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "bdd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]"]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
        ]
    elif args.dataset == "sim10k":
        print("loading our dataset...........")
        args.s_imdb_name = "sim10k_2019_train"
        args.t_imdb_name = "cityscapes_car_2019_train"
        args.s_imdbtest_name = "sim10k_2019_val"
        args.t_imdbtest_name = "cityscapes_car_2019_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4,8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "mskda_car":
        print("loading our dataset...........")
        args.s1_imdb_name = "cityscapes_ms_car_train"
        args.s2_imdb_name = "KITTI_car_train"
        args.s1_imdbtest_name = "cityscapes_ms_car_test"
        args.s2_imdbtest_name = "KITTI_car_test"
        args.t_imdb_name = "bdd100k_daytime_car_train"
        args.t_imdbtest_name = "bdd100k_daytime_car_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    args.cfg_file = (
        project_path + "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else project_path + "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    s1_imdb, s1_roidb, s1_ratio_list, s1_ratio_index = combined_roidb(args.s1_imdb_name)
    s1_train_size = len(s1_roidb)  # add flipped         image_index*2

    s2_imdb, s2_roidb, s2_ratio_list, s2_ratio_index = combined_roidb(args.s2_imdb_name)
    s2_train_size = len(s2_roidb)  # add flipped         image_index*2

    t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
    t_train_size = len(t_roidb)  # add flipped         image_index*2

    load_name = args.model_dir

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            t_imdb.classes,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=args.class_agnostic,
        )

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    # print(fasterRCNN.state_dict().keys())

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(
        {k: v for k, v in checkpoint["model"].items() if k in fasterRCNN.state_dict()}
    )

    # fasterRCNN.load_state_dict(checkpoint['model'])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = args.part + args.model_dir.split("/")[-1]
    num_images = len(t_imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(t_imdb.num_classes)]

    output_dir = get_output_dir(t_imdb, save_name)

    dataset_s1 = roibatchLoader(
        s1_roidb,
        s1_ratio_list,
        s1_ratio_index,
        1,
        s1_imdb.num_classes,
        training=True,
    )

    dataloader_s1 = torch.utils.data.DataLoader(
        dataset_s1,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    dataset_s2 = roibatchLoader(
        s2_roidb,
        s2_ratio_list,
        s2_ratio_index,
        1,
        s2_imdb.num_classes,
        training=True,
    )

    dataloader_s2 = torch.utils.data.DataLoader(
        dataset_s2,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    dataset_t = roibatchLoader(
        t_roidb,
        t_ratio_list,
        t_ratio_index,
        1,
        t_imdb.num_classes,
        training=True,
    )

    dataloader_t = torch.utils.data.DataLoader(
        dataset_t,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    data_iter_t = iter(dataloader_t)
    data_iter_s1 = iter(dataloader_s1)
    data_iter_s2 = iter(dataloader_s2)

    _t = {"im_detect": time.time(), "misc": time.time()}
    det_file = os.path.join(output_dir, "detections.pkl")

    point_number = 100
    features_sub1 = []
    features_sub2 = []
    features_ema = []

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    #for i in range(num_images):
    for i in range(point_number):
        misc_tic = time.time()
        data_s1 = next(data_iter_s1)
        data_s2 = next(data_iter_s2)
        data_t = next(data_iter_t)

        det_tic = time.time()
        with torch.no_grad():
            im_data.resize_(data_s1[0].size()).copy_(data_s1[0])
            im_info.resize_(data_s1[1].size()).copy_(data_s1[1])
            gt_boxes.resize_(data_s1[2].size()).copy_(data_s1[2])
            num_boxes.resize_(data_s1[3].size()).copy_(data_s1[3])
            source1 = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                tsne=True
            )
        with torch.no_grad():
            im_data.resize_(data_s2[0].size()).copy_(data_s2[0])
            im_info.resize_(data_s2[1].size()).copy_(data_s2[1])
            gt_boxes.resize_(data_s2[2].size()).copy_(data_s2[2])
            num_boxes.resize_(data_s2[3].size()).copy_(data_s2[3])
            source2 = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                tsne=True
            )

        with torch.no_grad():
            im_data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes.resize_(data_t[2].size()).copy_(data_t[2])
            num_boxes.resize_(data_t[3].size()).copy_(data_t[3])
            target = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                tsne=True
            )

        features_sub1.append(source1.view(-1).cpu().numpy())
        features_sub2.append(source2.view(-1).cpu().numpy())
        features_ema.append(target.view(-1).cpu().numpy())

        misc_toc = time.time()
        each_time = misc_toc - misc_tic

        sys.stdout.write(
            "im_detect: image {:d} {:.3f}s   \r".format(
                i + 1, each_time
            )
        )
        sys.stdout.flush()

    features_sub1 = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features_sub1)
    # = scale_to_01_range(features_sub1)

    features_sub2 = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features_sub2)
    #features_sub2 = scale_to_01_range(features_sub2)

    features_ema = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features_ema)
    #features_ema = scale_to_01_range(features_ema)

    x = features_sub1[ :, 0]
    y = features_sub1[ :, 1]
    plt.scatter(x, y, c="green")

    x = features_sub2[:, 0]
    y = features_sub2[:, 1]
    plt.scatter(x, y, c="blue")

    x = features_ema[:, 0]
    y = features_ema[:, 1]
    plt.scatter(x, y, c="red")
    plt.show()

    end = time.time()
    print("test time: %0.4fs" % (end - start))
