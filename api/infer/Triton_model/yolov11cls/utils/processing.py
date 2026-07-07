from api.infer.Utils.boundingbox import BoundingBox

import cv2
import numpy as np


def preprocess(img, input_shape, letter_box=True):
    """
    YOLOv11分类模型专用预处理函数（对齐官方逻辑）
    Args:
        img (np.ndarray): OpenCV读取的BGR格式图像（uint8）
        input_shape (tuple): 模型输入尺寸 (height, width)，默认(224,224)
    Returns:
        np.ndarray: 预处理后的CHW格式float32张量，已做ImageNet标准化
    """
    # 1. BGR→RGB（对齐官方）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 缩放：用INTER_LINEAR_EXACT（OpenCV4.5+）模拟antialias=True
    interpolation = cv2.INTER_LINEAR_EXACT if cv2.__version__ >= "4.5.0" else cv2.INTER_LINEAR
    img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=interpolation)

    # 3. 维度转换：HWC→CHW
    img = img.transpose((2, 0, 1))

    # 4. 仅归一化/255（无额外标准化，对齐官方）
    img = img.astype(np.float32) / 255.0

    return img


def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, label_names=[],
                letter_box=False):
    """Postprocess TensorRT outputs (分类模型专用).
    # Args
        output: TensorRT 分类模型输出（shape一般为 [1, 类别数]，即 [batch_size, num_classes]）
        origin_w: 原图宽度
        origin_h: 原图高度
        input_shape: 模型输入尺寸（如 [224,224]）
        conf_th: 置信度阈值
        label_names: 类别名称列表（顺序必须和训练时的类别顺序一致，如 ['wearing_suit', 'other_images']）
    # Returns
        list of BoundingBox: 分类结果（含classID、置信度、类别名等）
    """

    # 1. 提取模型输出并简化张量（适配分类模型输出格式）
    # 分类模型输出一般为 [batch_size, num_classes]，挤压后得到 [num_classes]
    model_output = output
    # 处理可能的批量维度（如 batch=1 时，从 [1, num_classes] → [num_classes]）
    class_probs = model_output.squeeze()  # 最终为一维数组：[类别0概率, 类别1概率, ..., 类别N概率]

    # 2. 找到最高概率的类别（核心：classID 就是这个类别索引）
    max_prob_idx = np.argmax(class_probs)  # 最大概率对应的索引 → 这就是 classID！
    classID = max_prob_idx  # 直接将索引指定为 classID（和训练时类别顺序一一对应）
    max_prob = class_probs[classID]  # 最高概率（置信度）
    # 获取类别名称（需确保 label_names 顺序和训练时一致）
    predicted_class_en = label_names[classID] if classID < len(label_names) else f"class_{classID}"

    # 3. 置信度过滤（低于阈值则返回空列表）
    if max_prob >= conf_th:
        # 分类场景无真实边界框，用 (0,0,1,1) 占位（不影响分类结果）
        return [BoundingBox(
            classID=classID,  # 最终指定的 classID
            confidence=float(max_prob),  # 置信度（转float避免numpy类型问题）
            x1=0, y1=0, x2=1, y2=1,  # 占位边界框
            image_width=origin_w, image_height=origin_h,
            classname=predicted_class_en
        )]
    else:
        return []