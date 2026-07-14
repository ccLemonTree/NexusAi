# -*- coding: utf-8 -*-
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.Utils.result_utils import LabelToModel
from api.infer.Utils.utils import Cos_Similarity
from tools.init import cfg
import struct,os
from tools.init import client
'''
人脸识别


'''
class Model(ModelClass):
    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][1]['label']
            models ,modelConf= LabelToModel(labels, cfg.unlogicModelDict)
            for i, info in enumerate(self.logicResult):
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                for aly, value in models.items():
                    bounding = self.tritonServer.run(aly, cut_img, self.cameraInfo, label_to_detect=value,
                                                          box_info=info)

                    if len(bounding)== 0:
                        continue

                    for sonbounding in bounding:
                        facevector = np.array(sonbounding.parames_vector, np.float32)
                        search_data = client.search(
                            collection_name=os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"),
                            data=[facevector],
                            anns_field='face_vector',
                            filter=None,
                            limit=1,
                            output_fields=["uuid"]
                        )

                        sonbounding.classname = self.logicModelName
                        distance = search_data[0][0]['distance']
                        if distance > self.conf:
                            sonbounding.x1 = info.x1
                            sonbounding.x2 = info.x2
                            sonbounding.y1 = info.y1
                            sonbounding.y2 = info.y2
                            sonbounding.image_width = info.image_width
                            sonbounding.image_height = info.image_height
                            sonbounding.u1 = info.u1
                            sonbounding.u2 = info.u2
                            sonbounding.v1 = info.v1
                            sonbounding.v2 = info.v2
                            sonbounding.parames_vector = {
                                "face_Info":{
                                    "face_id": search_data[0][0]['entity']['uuid'],
                                    "face_conf": round(distance,2)
                                }
                            }
                            boundingboxs.append(sonbounding)
            return boundingboxs,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)
        finally:
            pass
        return [],[]
