import time

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.retina.utils.processing import *
from api.infer.Triton_model.retina.utils.postprocess import  bounding_decode


def retina(triton_client, service_name, init_data, img, label_rules, box_info, conf_thres=0.4, iou_thres=0.3, conf=None):
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
    image_height,image_width,channle = img.shape
    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))

    # 修改此处 预处理
    input_image_buffer = preprocess(img, input_shape)
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
    dets, landms = bounding_decode(
        img,
        output_data[0],  # 位置输出
        output_data[1],  # 置信度输出
        output_data[2],  # 关键点输出
        0.2,
        iou_thres
    )
    label_name = label_names[0]
    for box,landms in zip(dets,landms):
        box= np.array(box[:4],dtype=int)
        x1,y1,x2,y2 = box
        conf=np.clip(box[:4],0,1,dtype=float)[0]
        landms = np.array(landms,dtype=int).tolist()
        if conf>=(label_rules[label_name]['conf'] if isinstance(label_rules,dict) else conf_thres):
            result_to_return.append(BoundingBox(0,conf,x1,x2,y1,y2,image_width,image_height,"face",parames_vector={'landms':landms}))

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return result_to_return, time_json
