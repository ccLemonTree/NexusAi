import time

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.plate.utils.utils import *

def plate_det(triton_client, service_name, init_data, img, label_rules, box_info, conf_thres=0.4, iou_thres=0.3, conf=None):
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
                if iou_thres == -1:
                    iou_thres = iou_thres
                else:
                    iou_thres = value["iou_thres"]
                if conf_thres == -1:
                    conf_thres = conf_thres
                else:
                    conf_thres = value["conf_thres"]

                model_version = value["model_version"]
                label_names = value["label_names"]

                for input in INPUT_DATA:
                    input_shape.append(input["dims"][0])
                    input_shape.append(input["dims"][1])
                    input_shape.append(input["dims"][2])
                    input_shape.append(input["dims"][3])
    inputs = []
    outputs = []
    output_data = []
    height,width,channel = img.shape
    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))

    # 修改此处 预处理
    input_image_buffer, r, left, top = detect_pre_precessing(img, input_shape[-2:])  # 检测前处理
    # input_image = input_image_buffer.astype(np.float32)
    inputs[0].set_data_from_numpy(input_image_buffer)

    pretreatment_end = time.time()  # 时间测试

    if triton_client.is_model_ready(service_name, model_version):
        results = triton_client.infer(model_name=service_name,
                                      inputs=inputs,
                                      outputs=outputs,
                                      model_version=model_version,
                                      client_timeout=30000)
        for output in OUTPUT_DATA:
            results.as_numpy(output["name"])
        for obj in OUTPUT_DATA:
            output_data.append(results.as_numpy(obj["name"]))

    result_to_return = []
    postprocessing_start = time.time()  # 时间测试
    if len(output_data) == 0:
        postprocessing_end = time.time()  # 时间测试
        time_json["Pretreatment"] = pretreatment_end - pretreatment_start
        time_json["post_time"] = postprocessing_start - pretreatment_end
        time_json["processing_time"] = postprocessing_end - postprocessing_start
        return result_to_return, time_json

    # 修改此处  后处理  套用此结构 detected_objects = [ BoundingBox(label, score, box[0], box[2], box[1], box[3], origin_w, origin_h,label_names[label])]
    outputs = post_precessing(output_data[0], r, left, top)  # 检测后处理
    detected_objects = []
    for output in outputs:
        rect = output[:4].tolist()
        label = int(output[-1])
        score = output[4]
        land_marks = output[5:13].reshape(4, 2).tolist()
        detected_objects.append(BoundingBox(0, score,rect[0],rect[2],rect[1],rect[3],width,height,"plate",parames_vector={"land_marks":land_marks,"plate_type":label}))

    for i in range(len(detected_objects)):
        box = detected_objects[i]
        label_name = label_names[box.classID]
        if label_name in label_rules:
            if box.confidence > (label_rules[label_name]['conf'] if isinstance(conf_thres, dict) else conf_thres):
                result_to_return.append(box)

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return result_to_return, time_json
