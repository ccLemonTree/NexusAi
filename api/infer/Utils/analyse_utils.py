# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2022年10月31日 11:15:15
@packageName Triton
@className analyse_utis
@version 1.0.0
@describe TODO
"""
import time
from api.infer.Utils.class_info import *
import numpy as np
import cv2
from tools.init import cfg

# Roi 点集格式的判断
def Roi_cvpoints(points, width, height):
    roi_points = []
    for point in points:
        roi_points.append([[
            int(round(float(point['x']) * width, 0)),
            int(round(float(point['y']) * height, 0))
        ]])
    roiRegion = np.array(roi_points, dtype=np.int32)
    return roiRegion

def Extract_roi(rois, points, img=None):
    """判断点 是否在 roi 点集中（1在 0边上 -1不在），True 在，False 不在"""
    flag = False
    if len(rois) != 0:
        for roi in rois:
            flag = True if cv2.pointPolygonTest(roi, points, True) > 0 else False
            if flag:
                break
    return flag


def Extract_nroi(nrois, points, img=None):
    """判断点是否在 nroi 点集外，不在返回 True，在返回 False"""
    flag = True
    if len(nrois) != 0:
        for nroi in nrois:
            flag = False if cv2.pointPolygonTest(nroi, points, True) > 0 else True
            if flag == False:
                break
    return flag


# 事件类型 找标签  0->smoke
def event_tolabel(events, eventcfg):
    """
    事件 找 标签
    :param events:  149
    :param eventcfg: cfg.load_eventcfg()
    :return:   找模型
    """
    value = eventcfg.get(events, None)
    if value is None:
        return []
    return value['label']


# 标签找模型 dog -> yolov7_dog
def LabelToModel(label:set, modelCfg:dict):
    """
    :param label :list
    :param modelCfg :dict
    """
    modelDict = {}
    modelConf = {}
    for lab in label:
        for model,value in modelCfg.items():
            if lab in value["classes"]:
                modelValue = modelDict.get(model,[])
                modelValue.append(lab)
                modelDict[model] = modelValue
                modelConf[lab] = {
                    'conf':value['conf'],
                    'iou':value['iou']
                }
    return modelDict,modelConf

# Roi 判断 points in Roi
def Roi_sorts(roi_list, width, height):
    """

    :param roi_list:   ROI 列表
        {
            "detectAreaId": 288197596961308693,
            "alarmTypeId": 2,
            "check": true,
            "points": [{
                "x": 0.0,
                "y": 0.0
                ....
            }]
        }

    :param width:      图片宽度
    :param height:     图片高度
    :return: [Analyse_Info object, ... ]

    结构体 内容
        Analyse_Info object:
            detectAreaId  ROI 标识
            alarmTypeId   事件类型
            roi = []      roi 区域
            nroi = []     nroi区域

    """
    roi_dict = []
    for roi in roi_list:
        analyse_info = AnalyseInfo()
        analyse_info.detectAreaId = roi["detectAreaId"]
        analyse_info.alarmTypeId = roi.get('alarmTypeId',99999)
        # 如果 flag 为 True 则为带逻辑的分析 roi中没有labels
        if roi.get("labels",None) is None:
            analyse_info.labels = event_tolabel(analyse_info.alarmTypeId,cfg.eventDict)
        else:
            analyse_info.labels = roi["labels"]
        roi_points = roi["points"]
        # 判断检测区域还是非检测区域
        analyse_info.flagRoi = roi["check"]
        # roi点集 格式化
        analyse_info.roi = [Roi_cvpoints(roi_points, width, height)]
        roi_dict.append(analyse_info)
    return roi_dict


# 计算时间的装饰器
def analyseTime(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()
        return result, stop - start

    return inner
