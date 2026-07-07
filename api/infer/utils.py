# -*- coding: utf-8 -*-
# @Time    : 2025/6/9 16:00:40
# @Author  : 陈澔麟
# @File    : utils.py

from PIL import Image
import os,uuid
import urllib.request as request
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


def image_file_to_base64(image_path, format='JPEG'):
    """
    将图片文件转换为Base64编码字符串

    参数:
        image_path: 图片文件路径
        format: 转换后的图片格式，如'JPEG'、'PNG'等

    返回:
        Base64编码字符串（带前缀，如'data:image/jpeg;base64,...'）
    """
    try:
        # 使用PIL读取图片
        with Image.open(image_path) as img:
            # 创建字节流
            buffer = BytesIO()
            # 保存图片到字节流（自动处理格式）
            img.save(buffer, format=format)
            # 转换为Base64编码
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # 添加数据前缀
            return f'data:image/{format.lower()};base64,{base64_str}'
    except Exception as e:
        print(f"图片文件转Base64失败: {str(e)}")
        return None


def numpy_to_base64(numpy_array, format='JPEG'):
    """
    将NumPy数组（OpenCV格式的图像）转换为Base64编码字符串

    参数:
        numpy_array: 图像的NumPy数组（BGR格式，OpenCV默认）
        format: 转换后的图片格式，如'JPEG'、'PNG'等

    返回:
        Base64编码字符串（带前缀，如'data:image/jpeg;base64,...'）
    """
    try:
        # OpenCV默认是BGR格式，需要转为RGB才能正确保存
        if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
            rgb_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_array = numpy_array  # 灰度图无需转换

        # 创建PIL图像
        img = Image.fromarray(rgb_array)

        # 保存到字节流
        buffer = BytesIO()
        img.save(buffer, format=format)

        # 转换为Base64编码
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # 添加数据前缀
        return f'data:image/{format.lower()};base64,{base64_str}'
    except Exception as e:
        print(f"NumPy数组转Base64失败: {str(e)}")
        return None



def file2base64img(file,exp_shape=640,prefix=True):
    try:
        # 获取图像的宽度和高度
        img = Image.open(file)
        # w, h = img.size
        # # 计算缩放比例
        # scale = exp_shape / w
        # # 计算新的宽度和高度
        # new_width = int(w * scale)
        # new_height = int(h * scale)
        # 生成随机文件名
        random_uuid = uuid.uuid4()
        filename = f"file_{random_uuid}.png"
        # 调整图像大小
        # resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # 保存调整大小后的图像
        img.save(filename)
        # 以二进制模式打开文件并读取内容
        with open(filename, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        # 将编码后的字节数据转换为字符串
        encoded_image_text = encoded_image.decode("utf-8")
        # 构建完整的Base64数据URL
        if prefix:
            base64_qwen = "data:image;base64,"+f"{encoded_image_text}"
        else:
            base64_qwen = f"{encoded_image_text}"

        # 删除临时文件
        os.remove(filename)
    except Exception as e:
        print(e)
        os.remove(filename)
        # if os.path.exists(filename):
        #     os.remove(filename)
        return None
    return base64_qwen

def siou(box1, box2):
    intersection_width = min(box1.x2, box2.x2) - max(box1.x1, box2.x1)
    intersection_height = min(box1.y2, box2.y2) - max(box1.y1, box2.y1)
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    box1_area = box1.width() * box1.height()
    box2_area = box2.width() * box2.height()
    iou = intersection_area / min(box1_area, box2_area)  # float(box1_area + box2_area - intersection_area)
    return iou

def imgRead(imgsList,picType):
    img = None
    images_list = []
    for fileName in imgsList:
        if picType=="url":
            response = request.urlopen(fileName)
            img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
        elif picType=="base64":
            imgData = base64.b64decode(fileName)
            nparr = np.fromstring(imgData, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif picType=='file':
            img = cv2.imread(fileName)
        images_list.append(img)
    return images_list