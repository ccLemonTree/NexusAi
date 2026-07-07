# -*- coding: utf-8 -*-
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.Utils.result_utils import LabelToModel
from api.infer.Utils.utils import Cos_Similarity
from tools.init import cfg
import struct
'''
人脸识别 


'''
class Model(ModelClass):
    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][1]['label']
            models = LabelToModel(labels, cfg.unlogicModelDict)
            for i, info in enumerate(self.logicResult):
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                for aly, value in models.items():
                    bounding = self.tritonServer.run(aly, cut_img, self.cameraInfo, label_to_detect=value,
                                                          box_info=info)
                    if len(bounding)== 0:
                        continue
                    bounding = bounding[0]
                    bounding.x1 = info.x1
                    bounding.x2 = info.x2
                    bounding.y1 = info.y1
                    bounding.y2 = info.y2
                    bounding.y2 = info.y2
                    bounding.classname = self.logicModelName
                    boundingboxs.append(bounding)
            return boundingboxs,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)
        finally:
            pass
        return [],[]


