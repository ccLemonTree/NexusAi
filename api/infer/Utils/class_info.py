# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2023年03月22日 14:10:22
@packageName Triton
@className class_info
@version 1.0.0
@describe TODO

ROI
"""

# 设备相关信息
import time
from tools.init import cfg


class CameraInfo:
    deviceId = None     # ANC0001
    presetId = 0        # 0
    fileName1 = "None"    # 第一张抓图 的完整目录
    fileName2 = "None"    # 第二张抓图 的完整目录
    baseName = "None"     # 第一张抓图的文件名称
    baseNameSplit = "None" # 去除.jpeg .png .jpg 等 后缀的 文件名称
    defaultLabels = ""    #  默认分析标签
    picType = 0         #  图片类型
    analyseROI = None   # roi
    detectAreaId = None  # 默认 区域id
    alarmTypeId = 0      # 默认分析类型
    reserveParam = None
    snap=0
    dirName = 0
    imgsList = []
    analyseStatus = '0'
    datetime = str(time.time())
    ptz = {'p':0,'z':0,'t':0}
# 事件类型结构体
class AnalyseInfo():
    """
    detectAreaId  ROI 标识
    alarmTypeId   事件类型
    roi = []      roi 区域
    nroi = []     nroi区域
    labels = []   labels 区域

    """
    detectAreaId = None
    alarmTypeId = None
    labels = []
    flagRoi = True
    roi = []


class ModelClass(object):
    def __init__(self,imgs,logicResults,cameraInfo,modelname,tritonServer,label_rules):
        self.picture = imgs[0]
        self.picture_2 = imgs[0] if len(imgs)==1 else imgs[1]
        self.cameraInfo = cameraInfo
        self.logicModelName = modelname
        self.tritonServer = tritonServer
        self.labelRules = label_rules
        self.cfg = cfg
        self.conf = self.cfg.logicModelDict[self.logicModelName]['param'][0]['conf'] if self.labelRules.get(self.logicModelName,None) is None else self.labelRules[self.logicModelName]['conf']
        self.iou = self.cfg.logicModelDict[self.logicModelName]['param'][1]['iou'] if self.labelRules.get(self.logicModelName,None) is None else self.labelRules[self.logicModelName]['iou']
        self.logicResult = [i for i in logicResults if i.confidence > self.conf]
    def pic2base64(self,image):
        import cv2
        import base64
        # 将图像转换为JPEG格式，然后编码为Base64
        _, buffer = cv2.imencode('.jpeg', image)
        jpg_as_text = base64.b64encode(buffer)
        base64_str = jpg_as_text.decode('utf-8')
        return base64_str
    def execute(self):
        pass


