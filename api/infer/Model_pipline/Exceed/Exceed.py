# -*- coding: utf-8 -*-
#人脸
import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from sklearn.cluster import DBSCAN
class Model(ModelClass):
    def siou(self, box1, box2):
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
        if len(self.logicResult)==0:
            return [] ,[]
        return2return = []
        try:
            group = {}
            silhouette_scores = []
            height,width,channle = self.picture.shape
            centers = [info.center_normalized() for info in self.logicResult]
            dbscan = DBSCAN(eps=0.15, min_samples=2)  # eps是邻域的半径，min_samples是最小样本数
            # 使用DBSCAN进行聚类
            labels = dbscan.fit_predict(centers)
            for index,value in enumerate(labels):
                lists = group.get(value,[])
                lists.append(self.logicResult[index])
                group[value] = lists
            if -1 in group.keys():del group[-1]
            for j,value in group.items():
                if len(value)>=self.labelRules[self.logicModelName]["num"]:
                    points = []
                    for j in value:
                        points.append([[j.x1,j.y1]])
                        points.append([[j.x2, j.y1]])
                        points.append([[j.x1, j.y2]])
                        points.append([[j.x2, j.y2]])
                    x,y,w,h = cv2.boundingRect(np.array(points))
                    return2return.append(BoundingBox(0, float(1.00), x,x+w,y,y+h,width,height ,self.logicModelName))
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)
        finally:
            return return2return,[]
