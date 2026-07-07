
import cv2
import numpy as np
from api.infer.Triton_model.triton_client import triton_inference
import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)  # override=True确保.env覆盖系统环境变量
from tools.init import cfg

if __name__ == '__main__':
    tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"), "api/infer/Triton_model/weights"),
                                    urls=[os.getenv("TRITON_SERVER")])
    service_name = "yolov5_helmet"
    images_path = r"example/ljz.png"
    classes = "fire_and_smoke" #cfg.unlogicModelDict[service_name]['classes']
    dete_dit = {}
    for i in classes:
        dete_dit[i] = {"iou":0.2,"conf":0.1}
    result_to_return = tritonServer.run(service_name, images_path, {'filename': '111'},dete_dit)
    for i in result_to_return:
        print(i)
