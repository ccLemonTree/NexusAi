import copy
import time

import cv2
import tritonclient.grpc as grpcclient
import numpy as np
# import torch
from Triton_model.pulcls.utils.process import VehicleAttribute
from Utils.boundingbox import BoundingBox
def softmax( f ):
    # 坏的实现: 数值问题
    return np.exp(f) / np.sum(np.exp(f))


def pulc(triton_client, service_name, init_data, img, label_to_detect, box_info, conf_thres=-1, iou_thres=-1):
    pretreatment_start = time.time()  # 时间测试
    time_json = {
        "filename": "",
        "model": "",
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }
    time_json["model"] = service_name

    model_version = ""
    INPUT_DATA = []
    OUTPUT_DATA = []
    label_names = []
    input_shape = []
    if not bool(init_data):
        print("模型信息没有读到")
    else:
        for key, value in init_data.items():
            if key == service_name:
                INPUT_DATA = value["input"]
                OUTPUT_DATA = value["output"]
                model_version = value["model_version"]
                label_names = value["label_names"]
                for input in INPUT_DATA:
                    input_shape.append(input["dims"][-2])  #192
                    input_shape.append(input["dims"][-1])  #256
                break
    inputs = []
    outputs = []
    det_obj = []
    IMAGES = img
    result_to_return = []
    h,w,c = IMAGES.shape
    IMAGES = cv2.resize(IMAGES, input_shape[::-1])
    IMAGES = cv2.cvtColor(IMAGES, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    IMAGES = IMAGES / 255
    IMAGES = (IMAGES - mean) / std
    IMAGES = np.transpose(IMAGES,(2, 0, 1))
    # IMAGES = to_tensor(IMAGES)
    IMAGES = [np.array([IMAGES])]
    ztclabel = VehicleAttribute()
    for input, input_image in zip(INPUT_DATA, IMAGES):
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))

    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))

    input_image = IMAGES[0].astype(np.float32)

    inputs[0].set_data_from_numpy(input_image)

    pretreatment_end = time.time()  # 时间测试
    # 检测模型服务是否正常 Ready
    if triton_client.is_model_ready(service_name, model_version):
        results = triton_client.infer(model_name=service_name,
                                      inputs=inputs,
                                      outputs=outputs,
                                      model_version=model_version,
                                      client_timeout=30000)
        for output in OUTPUT_DATA:
            results.as_numpy(output["name"])
        for obj in OUTPUT_DATA:
            det_obj.append(results.as_numpy(obj["name"]))
    postprocessing_start = time.time()  # 时间测试
    output = det_obj[0]
    result = ztclabel(output)
    print(result)
    info = copy.deepcopy(box_info)
    info.parames_vector = result[0]
    info.classname = 'ztc_mutil'
    # info.x1 = 0
    # info.x2 = 0
    # info.y1 = 0
    # info.y2 = 0
    # result_to_return.append(BoundingBox(0,'0.9',0,100,0,100,w,h,label_to_detect[0],parames_vector=info))
    return [info], time_json
