
import cv2
import numpy as np

from api.infer.Utils.boundingbox import BoundingBox


def preprocess_image(img):
    """预处理图像（修正数据类型为float32）"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    img = cv2.resize(img, (150, 150))  # 缩放为模型要求的尺寸
    img = img.astype(np.float32)  # 转换为float32类型（关键修正）
    img = img / 255.0  # 归一化到0-1范围（符合大多数深度学习模型的输入要求）
    return np.expand_dims(img, axis=0)  # 增加批次维度，形状为[1, 150, 150, 3]


def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5,label_names=[], letter_box=False):
    return [BoundingBox(0,0.99,100,101,100,101,origin_w,origin_h,label_names[0],parames_vector=output.tolist()[0])]
