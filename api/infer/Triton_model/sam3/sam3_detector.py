# -*- coding: utf-8 -*-
"""
@File    : sam3_detector.py.py
@Time    : 2025/12/15 16:44
@Author  : 陈冲
@Description : sam3推理
@Version : 1.0
"""

import time
from PIL import Image
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.sam3.utils.processing import *
import numpy as np


def sam3(triton_client, service_name, init_data, img, label_rules, box_info, conf_thres=-1, iou_thres=-1, conf=None):
    pretreatment_start = time.time()  # [时间点1] 预处理开始
    # --- 1. 图像处理 ---
    if isinstance(img, np.ndarray):
        orig_h, orig_w = img.shape[:2]
        img = Image.fromarray(img[..., ::-1])
    else:
        orig_w, orig_h = img.size

    img_resized = img.convert('RGB').resize((1008, 1008))
    # img_data = (np.array(img_resized).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    img_data = np.array(img_resized).astype(np.uint8).transpose(2, 0, 1)[None]

    # --- 2. 初始化 time_json ---
    time_json = {
        "filename": "",
        "model": service_name,
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }

    # --- 3. 解析配置 ---
    final_conf_thres = 0.5
    INPUT_CFG = []
    OUTPUT_CFG = []

    if init_data and service_name in init_data:
        cfg = init_data[service_name]
        INPUT_CFG = cfg.get("input", [])
        OUTPUT_CFG = cfg.get("output", [])
        if conf_thres == -1:
            final_conf_thres = cfg.get("conf_thres", 0.5)
        else:
            final_conf_thres = conf_thres
    else:
        if conf_thres != -1: final_conf_thres = conf_thres

    if len(INPUT_CFG) < 2:
        return [], time_json

    # --- 4. 构造 Prompt ---
    label_list_analyse = list(label_rules.keys()) if isinstance(label_rules, dict) else list(label_rules)
    label_list = [i.replace("sam3_","") for i in label_list_analyse]
    TEXT_PROMPTS = np.array([str(x).encode('utf-8') for x in label_list], dtype=object)

    # --- 5. 构造 Triton Inputs ---
    img_in_name = INPUT_CFG[0]["name"]
    txt_in_name = INPUT_CFG[1]["name"]
    img_type = INPUT_CFG[0]["data_type"].split("_")[-1]
    txt_type = INPUT_CFG[1]["data_type"].split("_")[-1]
    if "STRING" in txt_type: txt_type = "BYTES"

    inputs = [
        grpcclient.InferInput(img_in_name, img_data.shape, img_type),
        grpcclient.InferInput(txt_in_name, TEXT_PROMPTS.shape, txt_type)
    ]
    inputs[0].set_data_from_numpy(img_data)
    inputs[1].set_data_from_numpy(TEXT_PROMPTS)

    outputs = [grpcclient.InferRequestedOutput(obj["name"]) for obj in OUTPUT_CFG]

    pretreatment_end = time.time()  # [时间点2] 预处理结束

    # --- 6. 推理 ---
    output_dict = {}
    if triton_client.is_model_ready(service_name):
        results = triton_client.infer(service_name, inputs, outputs=outputs)
        for obj in OUTPUT_CFG:
            name = obj["name"]
            data = results.as_numpy(name)
            output_dict[name] = data

    postprocessing_start = time.time()  # [时间点3] 后处理开始

    # --- 7. 结果判空与后处理 ---
    detected_objects = []

    if not output_dict:
        postprocessing_end = time.time()
        time_json["Pretreatment"] = pretreatment_end - pretreatment_start
        time_json["post_time"] = postprocessing_start - pretreatment_end
        time_json["processing_time"] = postprocessing_end - postprocessing_start
        return detected_objects, time_json

    # 调用 postprocess 时传入 label_rules
    detected_objects = postprocess(
        output_dict,
        orig_w,
        orig_h,
        final_conf_thres,
        label_list_analyse,
        label_rules=label_rules  # 传入规则字典
    )

    postprocessing_end = time.time()  # [时间点4] 后处理结束

    # --- 8. 时间计算 ---
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start

    return detected_objects, time_json
