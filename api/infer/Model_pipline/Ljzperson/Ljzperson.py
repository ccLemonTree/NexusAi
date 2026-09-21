# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 17:51
# @Author  : 陈澔麟
# @File    : Ljzperson.py
# -*- coding: utf-8 -*-
#人脸
import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun

class Model(ModelClass):
    def Liou(self,box1, box2):
        '''
        将符号坐标和模板坐标放入IOU进行匹配
        :param box1: 符号坐标列表
        :param box2: 模板坐标列表
        :return: 交并比结果 题目和符号重合的地方占整个符号的多少
        '''
        intersection_width = min(box1.x2, box2.x2) - max(box1.x1, box2.x1)
        intersection_height = min(box1.y2, box2.y2) - max(box1.y1, box2.y1)
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0
        intersection_area = intersection_width * intersection_height
        box1_area = box1.width() * box1.height()
        box2_area = box2.width() * box2.height()

        iou = intersection_area / min(box1_area, box2_area)  # float(box1_area + box2_area - intersection_area)
        return iou

    def execute(self):
        try:
            height,width,channle = self.picture.shape
            k = 0
            appends = []
            redx = []
            param1 = cfg.logicModelDict[self.logicModelName][1]['label'][0]
            for info in self.logicResult:
                if info.classname == param1:
                    for indexss,dbinfo in enumerate(self.logicResult):
                        if dbinfo.classname != param1:
                            dbinfo.x1 -= 150
                            dbinfo.y1 -= 150
                            dbinfo.x2 += 150
                            dbinfo.y2 += 150
                            iou = self.Liou(dbinfo,info)
                            # cv2.rectangle(self.picture,(dbinfo.x1-150,dbinfo.y1-150),(dbinfo.x2+150,dbinfo.y2+150),(255,0,0),1,1,1)
                            # cv2.rectangle(self.picture,(info.x1,info.y1),(info.x2,info.y2),(255,255,0),1,1,1)
                            # cv2.imshow("1",self.picture)
                            # cv2.waitKey(0)
                            # 如果有iou存在就 记录垃圾堆的报警索引
                            if iou!=0.0:
                                redx.append(indexss)
            #如果没有重叠就报警
            if len(redx)==0:
                for infos in self.logicResult:
                    if infos.classname != param1:
                        infos.classname = self.logicModelName
                        appends.append(infos)
                return appends,[]
            # 如果有重叠 对索引排序 从大的开始删
            redx = list(set(redx))
            redx.reverse()
            # 删除重叠目标
            for i in redx:    
                self.logicResult.pop(i)
            for infos in self.logicResult:
                if infos.classname != param1:
                    infos.classname = self.logicModelName
                    appends.append(infos)
            return appends,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [],[]


