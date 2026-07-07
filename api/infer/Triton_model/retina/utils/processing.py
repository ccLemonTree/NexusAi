from api.infer.Utils.boundingbox import BoundingBox
import cv2
import numpy as np

def preprocess(image, input_shape=(1, 1000, 1000, 3)):
    """
    预处理图像以匹配模型输入。
    假设模型输入是 NHWC 格式的 uint8。
    返回预处理后的图像和缩放比例。
    """
    img_resized = cv2.resize(image, (640, 640))  # Resize the image to 640x640

    img = np.float32(img_resized)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    # 返回预处理后的图像、缩放比例和原始图像
    return img
