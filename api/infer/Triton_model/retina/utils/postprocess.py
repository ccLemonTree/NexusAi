import numpy as np
import cv2
import os
from api.infer.Utils.boundingbox import BoundingBox
from api.infer.Triton_model.retina.utils.utils import py_cpu_nms
from api.infer.Triton_model.retina.utils.config import cfg_re50
from api.infer.Triton_model.retina.utils.functions import PriorBox

import numpy as np


def decode(loc, priors, variances):
    """
    将预测的偏移量转换为边界框坐标
    参数:
        loc: 预测的边界框偏移量，形状为 [N, 4]
        priors: 先验框，形状为 [N, 4]，格式为 [x_center, y_center, width, height]
        variances: 方差，形状为 [2]
    返回:
        boxes: 解码后的边界框，形状为 [N, 4]，格式为 [x1, y1, x2, y2]
    """
    # 计算中心坐标：prior中心 + loc偏移量 * 方差[0] * prior宽高
    centers = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    # 计算宽高：prior宽高 * exp(loc宽高偏移 * 方差[1])
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])

    # 合并中心和宽高为 [x_center, y_center, width, height]
    boxes = np.concatenate([centers, sizes], axis=1)

    # 转换为 [x1, y1, x2, y2] 格式
    boxes[:, :2] -= boxes[:, 2:] / 2  # x1 = x_center - width/2, y1 = y_center - height/2
    boxes[:, 2:] += boxes[:, :2]  # x2 = x1 + width, y2 = y1 + height

    return boxes


def decode_landm(landm, priors, variances):
    """
    将预测的关键点偏移量转换为关键点坐标
    参数:
        landm: 预测的关键点偏移量，形状为 [N, 10]（5个点，每个点x、y坐标）
        priors: 先验框，形状为 [N, 4]，格式为 [x_center, y_center, width, height]
        variances: 方差，形状为 [2]
    返回:
        landms: 解码后的关键点，形状为 [N, 10]，格式为 [x1, y1, x2, y2, ..., x5, y5]
    """
    # 每个关键点坐标 = prior中心 + 对应偏移量 * 方差[0] * prior宽高
    landms = np.concatenate([
        priors[:, :2] + landm[:, :2] * variances[0] * priors[:, 2:],  # 第1个点
        priors[:, :2] + landm[:, 2:4] * variances[0] * priors[:, 2:],  # 第2个点
        priors[:, :2] + landm[:, 4:6] * variances[0] * priors[:, 2:],  # 第3个点
        priors[:, :2] + landm[:, 6:8] * variances[0] * priors[:, 2:],  # 第4个点
        priors[:, :2] + landm[:, 8:10] * variances[0] * priors[:, 2:]  # 第5个点
    ], axis=1)

    return landms

def bounding_decode(img_raw,loc,conf,landms,threshold,nms):
    priorbox = PriorBox(cfg_re50,image_size=(640, 640))
    priors = priorbox.forward()

    boxes = decode(loc[0], priors, cfg_re50['variance'])
    scores = conf[0][:, 1]
    landms = decode_landm(landms[0], priors, cfg_re50['variance'])

    # Scale boxes and landmarks to the original image size
    scale = np.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])
    boxes = boxes * scale
    landms_scale = np.array([img_raw.shape[1], img_raw.shape[0]] * 5)
    landms = landms * landms_scale

    # Filter boxes with a threshold
    inds = np.where(scores > threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # Apply NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms)
    dets = dets[keep, :]
    landms = landms[keep]
    return dets, landms