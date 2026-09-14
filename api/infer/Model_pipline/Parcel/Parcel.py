from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

class Model(ModelClass):

    @staticmethod
    def boxes_intersect(box1, box2):
        return (
            box1.x1 < box2.x2
            and box1.x2 > box2.x1
            and box1.y1 < box2.y2
            and box1.y2 > box2.y1
        )

    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][0]['label']
            parcels = [
                box for box in self.logicResult
                if box.classname == labels[-1] and box.width() >= 30 and box.height() >= 30
            ]
            for info in self.logicResult:
                if info.classname == labels[-1]:
                    continue
                if any(self.boxes_intersect(info, parcel) for parcel in parcels):
                    info.classname = self.logicModelName
                    boundingboxs.append(info)
            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []

