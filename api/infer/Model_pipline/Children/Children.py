# -*- coding: utf-8 -*-
"""
@File    : Children.py
@Time    : 2026/10/23 15:56
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""
from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

class Model(ModelClass):

    def execute(self):
        try:
            child_label = cfg.logicModelDict[self.logicModelName][0]['label'][0]
            if len(self.logicResult) == 1 and self.logicResult[0].classname == child_label:
                self.logicResult[0].classname = self.logicModelName
                return self.logicResult, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
