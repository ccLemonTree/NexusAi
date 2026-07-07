# -*- coding: utf-8 -*-
# 人脸
import os, time
import numpy as np
from tools.init import cfg
from api.infer.Utils.result_utils import LabelToModel
from api.infer.running import analyseRun
from api.infer.Utils.class_info import ModelClass
class Model(ModelClass):
    def execute(self) -> [list, dict]:
        boundingboxs = []
        logicResult_copy = []
        box_infos = []
        labels = cfg.logicModelDict[self.logicModelName][1]['label']
        # 第一次是 检测到自行车的boxs  给首次目标创建 父节点
        for info in self.logicResult:
            cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
            height, width, channle = self.picture.shape
            boundingboxs = analyseRun(labels, [cut_img, cut_img], self.cameraInfo, box_info=info)
            if len(boundingboxs) > 1:
                for boxinfo in boundingboxs:
                    if boxinfo.y1 < int(height * 0.30):
                        logicResult_copy.append(boxinfo)
            else:
                continue
            if len(logicResult_copy)>1:
                # 添加符合条件的框
                info.classname = self.logicModelName
                box_infos.append(info)
                        # 添加符合条件的框 的距离中心点的距离
                        # shuffleList.append(
                        #     self.calculate_distance_and_slope((boxinfo.center_boxs()), (picWidthCenter, boxinfo.y1), flag=True))

        if len(boundingboxs) > 1:
            return box_infos,[]
        return [],[]

    def __del__(self):
        """
        后处理  类注销前操作的 内容  或者 抛出 无法捕捉的异常报错  执行的内容
        :return:
        """
        pass
