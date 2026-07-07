import time
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun
from api.infer.qwen3_vl_reranker_client import Qwen3VLRerankerClient
import cv2
reranker_client = Qwen3VLRerankerClient()
class Model(ModelClass):
    def execute(self):
        try:
            boundingboxs = []
            for info in self.logicResult:
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                if len(self.logicResult) !=0:
                    rer_conf = cfg.logicModelDict[self.logicModelName]["param"][2]["rer_conf"]
                    text1 = cfg.logicModelDict[self.logicModelName]['rer_label'][0]
                    text2 = cfg.logicModelDict[self.logicModelName]['rer_label'][1]

                    logger.info(f"rer_conf: {rer_conf}")
                    logger.info(f"text1: {text1}")
                    logger.info(f"text2: {text2}")

                    start_time = time.time()
                    scores = reranker_client.rerank(cut_img, [text1, text2])
                    logger.info(f"重排序耗时：{time.time() - start_time}")

                    best_idx = int(np.argmax(scores))
                    best_text = text1 if best_idx == 0 else text2
                    best_score = float(scores[best_idx])

                    if best_idx != 0 or best_score <= rer_conf:
                        continue
                    logger.info(f"best_text: {best_text}| best_score: {best_score} ")
                    info.classname = self.logicModelName
                    boundingboxs.append(info)
            
            return boundingboxs, []

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []