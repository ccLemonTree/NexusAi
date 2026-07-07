# -*- coding: utf-8 -*-
#人脸
import cv2,os
import numpy as np
from Utils.class_info import ModelClass
from Utils.boundingbox import BoundingBox
from Utils.config import cfg
from Beta_Logging.log_utils import logger
from sklearn.cluster import DBSCAN
import pickle
from Utils.analyse_utils import Extract_roi
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



    def DBSCAN_filter(self, boundingbox):
        logicresult = []
        centers = [info.center_normalized() for info in boundingbox]
        dbscan = DBSCAN(eps=0.08, min_samples=2)  # eps是邻域的半径，min_samples是最小样本数
        # 使用DBSCAN进行聚类
        labels = dbscan.fit_predict(centers)
        for index,value in enumerate(labels):
            if value!=-1:
                logicresult.append(boundingbox[index])
        return logicresult

    def execute(self):
        '''
        预警生成条件
        1. 密度， 从上一次到下当前 密度有变化。且有一定的阈值。
        2. 聚集人数超过10 人


        '''
        return2return = []
        images_list = []
        if len(self.logicResult) == 0:
            return [], []

        deviceid = str(self.cameraInfo.deviceId)
        presetid = str(self.cameraInfo.presetId)
        base_path = os.path.join(cfg.root_path, "Picture_pkl", deviceid, presetid, self.logicModelName)
        if os.path.exists(base_path) == False:
            os.makedirs(base_path, exist_ok=True)
        if len(os.listdir(base_path)) == 0:
            return [], []
        try:
            group = {}
            silhouette_scores = []
            height,width,channle = self.picture.shape
            centers = [info.center_normalized() for info in self.logicResult]
            dbscan = DBSCAN(eps=0.1, min_samples=2)  # eps是邻域的半径，min_samples是最小样本数
            # 使用DBSCAN进行聚类
            labels = dbscan.fit_predict(centers)
            for index,value in enumerate(labels):
                lists = group.get(value,[])
                lists.append(self.logicResult[index])
                group[value] = lists
            if -1 in group.keys():del group[-1]
            ppicture = self.picture.copy()
            flag = False
            for j,value in group.items():
                inlist = []
                if len(value)>=cfg.logicModelDict[self.logicModelName]["param"]["num"]:
                    points = []
                    for j in value:
                        points.append([[j.x1,j.y1]])
                        points.append([[j.x2, j.y1]])
                        points.append([[j.x1, j.y2]])
                        points.append([[j.x2, j.y2]])
                    x,y,w,h = cv2.boundingRect(np.array(points))
                    with open(os.path.join(base_path, "save.pkl"), "rb") as f:
                        last_dat = pickle.load(f)
                        for point in last_dat:
                            isin = Extract_roi(np.array([[[x,y],[x,y+h],[x+w,y+h],[x+w,y]]],dtype=np.int32),point.center_boxs())
                            if isin:
                                inlist.append(point)
                    if (len(inlist)!=0) and (len(value) - len(inlist)>=cfg.logicModelDict[self.logicModelName]["param"]["num"]//2+1):
                        return2return.append(BoundingBox(0, float(1.00), x,x+w,y,y+h,width,height ,self.logicModelName))
                        images_list.append(cv2.imread(os.path.join(base_path, "picture.jpeg")))
                        flag = True
            if flag:
                return return2return, images_list
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)
        finally:
            # 聚集重新计算
            self.remove(os.path.join(base_path, "save.pkl"))
            self.remove(os.path.join(base_path, "picture.jpeg"))
            with open(os.path.join(base_path, "save.pkl"), "wb") as f:
                filter_logicResult = self.DBSCAN_filter(self.logicResult)
                pickle.dump(filter_logicResult, f)
            cv2.imwrite(os.path.join(base_path, "picture.jpeg"), self.picture)
