import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun


class Model(ModelClass):

    def is_intersect(self, box1, box2):
        """判断两个框是否有实际重叠面积"""
        return (
            max(box1.x1, box2.x1) < min(box1.x2, box2.x2)
            and max(box1.y1, box2.y1) < min(box1.y2, box2.y2)
        )

    def box_distance(self, box1, box2):
        """计算两个框之间的最小距离；如果已经相交，距离为 0"""
        dx = max(max(box1.x1, box2.x1) - min(box1.x2, box2.x2), 0)
        dy = max(max(box1.y1, box2.y1) - min(box1.y2, box2.y2), 0)
        return (dx ** 2 + dy ** 2) ** 0.5

    def get_distance_threshold(self, box1, box2):
        """
        根据目标框大小动态计算距离阈值。

        规则：
        1. 取两个框短边中较小的那个作为参考
        2. 阈值 = 参考长度 * 10%
        3. 最小不低于 10 像素
        4. 最大不超过 40 像素
        """
        w1 = box1.x2 - box1.x1
        h1 = box1.y2 - box1.y1
        w2 = box2.x2 - box2.x1
        h2 = box2.y2 - box2.y1

        base = min(w1, h1, w2, h2)
        return min(40, max(10, base * 0.1))

    def is_intersect_or_near(self, box1, box2):
        """判断两个框是否相交，或者距离足够近"""
        if self.is_intersect(box1, box2):
            return True

        distance = self.box_distance(box1, box2)
        threshold = self.get_distance_threshold(box1, box2)

        return distance <= threshold

    def has_intersect_or_near_result(self, info, result):
        """判断 info 是否和 result 中任意目标框相交或接近"""
        for target in result:
            if self.is_intersect_or_near(info, target):
                return True
        return False

    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][1]['label']

            for info in self.logicResult:
                result = analyseRun(
                    labels,
                    [self.picture, self.picture],
                    self.cameraInfo,
                    box_info=info
                )

                if len(result) != 0 and self.has_intersect_or_near_result(info, result):
                    info.classname = self.logicModelName
                    boundingboxs.append(info)

            return boundingboxs, []

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
