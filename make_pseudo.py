import torch
import numpy as np
from torch.autograd import Variable
from model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from random import randint
import cv2


def make_pseudo(model1,model2, data, im_data, im_info, gt_boxes, num_boxes, num_classes, set_cfg=False, confidence_score=0.5):
    if set_cfg:
        pred_num = int(set_cfg[-1])
    else:
        pred_num = 30

    num_classes += 1

    model1.eval()
    model2.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    div_data = data[1][0][2].item()


    with torch.no_grad():
        (
            rois1,
            cls_prob1,
            bbox_pred1,
            rpn_loss_cls1,
            rpn_loss_box1,
            RCNN_loss_cls1,
            RCNN_loss_bbox1,
            rois_label1,
            d_pixel1,
            domain_p1,
        ) = model1(im_data, im_info, gt_boxes, num_boxes)

        (
            rois2,
            cls_prob2,
            bbox_pred2,
            rpn_loss_cls2,
            rpn_loss_box2,
            RCNN_loss_cls2,
            RCNN_loss_bbox2,
            rois_label2,
            d_pixel2,
            domain_p2,
        ) = model2(im_data, im_info, gt_boxes, num_boxes)

    #all_boxes = [[[] for _ in range(1)] for _ in range(9)]
    all_boxes = [[[] for _ in range(1)] for _ in range(11)]

    ## model1 output
    scores1 = cls_prob1.data
    boxes1 = rois1.data[:, :, 1:5]

    box_deltas1 = bbox_pred1.data
    # pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

    box_deltas1 = (
            box_deltas1.view(-1, 4)
            * torch.FloatTensor((0.1, 0.1, 0.2, 0.2)).cuda()
            + torch.FloatTensor((0.0, 0.0, 0.0, 0.0)).cuda()
    )
    box_deltas1 = box_deltas1.view(1, -1, 4 * num_classes)

    pred_boxes1 = bbox_transform_inv(boxes1, box_deltas1, 1)
    pred_boxes1 = clip_boxes(pred_boxes1, im_info.data, 1)

    scores1 = scores1.squeeze()
    pred_boxes1 = pred_boxes1.squeeze()

    ## model2 output
    scores2 = cls_prob2.data
    boxes2 = rois2.data[:, :, 1:5]

    box_deltas2 = bbox_pred2.data
    # pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

    box_deltas2 = (
            box_deltas2.view(-1, 4)
            * torch.FloatTensor((0.1, 0.1, 0.2, 0.2)).cuda()
            + torch.FloatTensor((0.0, 0.0, 0.0, 0.0)).cuda()
    )
    box_deltas2 = box_deltas2.view(1, -1, 4 * num_classes)

    pred_boxes2 = bbox_transform_inv(boxes2, box_deltas2, 1)
    pred_boxes2 = clip_boxes(pred_boxes2, im_info.data, 1)

    scores1 = scores1.squeeze()
    pred_boxes1 = pred_boxes1.squeeze()

    scores2 = scores2.squeeze()
    pred_boxes2 = pred_boxes2.squeeze()

    ## merge all result
    scores = torch.cat((scores1,scores2),dim=0)
    pred_boxes = torch.cat((pred_boxes1,pred_boxes2),dim=0)

    for j in range(1, num_classes):
        inds = torch.nonzero(scores[:, j] > 0.0).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], 0.5)
            cls_dets = cls_dets[keep.view(-1).long()]
            all_boxes[j] = cls_dets.cpu().numpy()

        else:
            all_boxes[j][i] = empty_array
    pseudo_gt = []
    for i in range(1, num_classes):
        tmp = all_boxes[i]
        tmp = tmp[tmp[:, 4] > confidence_score]
        if len(tmp) > 0:
            tmp[:, 4] = int(i)
            if len(pseudo_gt) == 0:
                pseudo_gt = tmp
            else:
                pseudo_gt = np.concatenate((pseudo_gt, tmp), axis=0)

    pred_num_boxes = torch.full((1,), len(pseudo_gt))
    if len(pseudo_gt) > pred_num:
        while len(pseudo_gt) > int(set_cfg[-1]):
            delete_axis = randint(0, len(pseudo_gt) - 1)
            pseudo_gt = np.delete(pseudo_gt, delete_axis, axis=0)
        pred_num_boxes = torch.full((1,), len(pseudo_gt))

    if len(pseudo_gt) < pred_num:
        if len(pseudo_gt) == 0:
            pseudo_gt = np.zeros((1, 5))
        while len(pseudo_gt) < pred_num:
            zero_axis = np.zeros((1, 5))
            pseudo_gt = np.concatenate((pseudo_gt, zero_axis), axis=0)

    pseudo_gt = torch.from_numpy(pseudo_gt)
    pseudo_gt = pseudo_gt.unsqueeze(0)

    # print(pseudo_gt)
    #
    # img_origin = im_data.squeeze().permute(1,2,0).cpu()
    # img_origin = np.asarray(img_origin,dtype=np.uint8)
    # cv2.imshow("",img_origin)
    # cv2.waitKey()

    return pseudo_gt, pred_num_boxes
