# -*- coding: utf-8 -*-

from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun


class Model(ModelClass):
    def execute(self):
        try:
            typeid = ["蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳", "香港单层", "香港双层", "澳门单层",
                      "澳门双层", "黄牌双层", "未知车牌"]

            cartype = {
                "dwtruck": "小货车",
                "dwmidtruck": "中卡",
                "dwvan": "厢式货车",
                "dwztc": "渣土车",
                "dwzk": "重卡",
                "dwygc": "油罐车",
                "car": "小汽车"
            }
            boundingboxs = []
            height, width, channle = self.picture.shape
            labels = cfg.logicModelDict[self.logicModelName][1]['label']
            for info in self.logicResult:
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                boundingbox = analyseRun(labels, [cut_img, cut_img], self.cameraInfo, box_info=info)
                if boundingbox != []:
                    info.parames_vector = {"carinfo": {
                        "lpr_id": boundingbox[0].parames_vector["idcard"],
                        "lpr_type": "",
                        "car_id": cartype.get(info.classname, ""),
                        "car_type": ""
                    }}
                    info.classname = self.logicModelName
                    boundingboxs.append(info)
            return boundingboxs, []

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
