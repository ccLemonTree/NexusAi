# -*- coding: utf-8 -*-
"""
@File    : processing.py.py
@Time    : 2025/11/4 13:18
@Author  : 陈冲
@Description : fastvlm处理方法
@Version : 1.0
"""
import numpy as np
from PIL import Image
import cv2
from api.infer.Utils.boundingbox import BoundingBox


def img_preprocess(img, background_color):
    """将图像扩展为正方形并填充背景"""
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    width, height = pil_image.size
    if width == height:
        return pil_image
    elif width > height:
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_image.mode, (height, height), background_color)
        result.paste(pil_image, ((height - width) // 2, 0))
        return result


def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, label_names=None,
                letter_box=False):
    return [BoundingBox(0, 0.9, 0, 0, 0,0, origin_w, origin_h, 'fire_and_smoke', parames_vector={"idcard": output})]
