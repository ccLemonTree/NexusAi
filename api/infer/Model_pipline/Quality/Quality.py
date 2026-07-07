import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun

class Model(ModelClass):

    def execute(self):
        try:
            boundingboxs = []
            logger.info(f"第一次检测结果：{self.logicResult}")
            labels = cfg.logicModelDict[self.logicModelName][1]['label']
            if len(self.logicResult) !=0:
                result = self.logicResult[0]
                logger.info(result)
                result.classname = self.logicModelName
                boundingboxs.append(result)
            return boundingboxs,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [],[]

