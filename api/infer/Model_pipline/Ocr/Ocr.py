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
            # 注意：res 必须是【OCR识别结果列表】，不是你存好的{进入,离开}字典！
            # 确认：self.logicResult[0].parames_vector["idcard"] 在进入循环前，是ocr原始list
            res = self.logicResult[0].parames_vector["ocr_list"]
            enter_text = ""
            leave_text = ""

            # 遍历原始OCR结果（dict列表 / BoundingBox对象列表兼容）
            for item in res:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                else:
                    # BoundingBox 对象场景
                    txt = getattr(item, "text", "")

                if "进入" in txt:
                    enter_text = txt
                if "离开" in txt:
                    leave_text = txt

            # 容错：防止识别不到文本 split 报错
            enter_num = enter_text.split("：")[-1] if enter_text else ""
            leave_num = leave_text.split("：")[-1] if leave_text else ""

            # 覆盖写入
            self.logicResult[0].parames_vector["ocr_list"] = {
                "进入": enter_num,
                "离开": leave_num
            }

            # ✅ 修复关键：不要返回 BoundingBox 对象！根据上层需求调整返回值
            # 方案A：如果你本意返回logicResult整体（数组形式）
            boundingboxs = self.logicResult
            # 方案B：如果只需要第0条，套进列表，避免上层当成单个对象
            # boundingboxs = [self.logicResult[0]]

            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [],[]

