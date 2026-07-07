import os
import base64
import datetime
import uuid

import cv2
import requests
import numpy as np
from PIL import Image
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

def timestamp_to_datetime(timestamp: float) -> str:
    """
    将时间戳转换为 2025-05-06 00:00:00 格式的字符串
    
    Args:
        timestamp: 时间戳（浮点型，单位为秒）
    
    Returns:
        格式化后的日期字符串
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def datetime_to_timestamp(date_str: str) -> float:
    """
    将 2025-05-06 00:00:00 格式的字符串转换为时间戳
    
    Args:
        date_str: 日期字符串（格式：YYYY-MM-DD HH:MM:SS）
    
    Returns:
        对应的时间戳（浮点型，单位为秒）
    """
    return int(datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())


def calculate_iou(box1, box2):
    """计算两个边界框的 IoU，格式 [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def img2base64(file):
    img = cv2.imread(file)
    h, w, c = img.shape
    scale_factor = 0.35
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    cv2.imwrite(file, cv2.resize(img, (new_width, new_height)))
    with open(file, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen


def cutimg2base64(img, type='obj'):
    random_uuid = uuid.uuid4()
    filename = fr"D:\inference\穹影智寻\static\bak\{type}\file_{random_uuid}.png"
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_image).save(filename)
    with open(filename, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen, filename


def vllm2embedding(data, type='img'):
    response = requests.post(
        "http://192.168.1.25:10015/v1/embeddings",
        json={
            "model": "gme-Qwen2-VL-2B-Instruct",
            "messages": [{
                "role": "user",
                "content": [

                    {"type": "image_url", "image_url": {"url": data}} if type == 'img' else {"type": "text",
                                                                                             "text": data}
                ],
            }],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()
    return response_json["data"][0]["embedding"]



def sglang2embedding(data, type='img'):
    url = "http://192.168.1.25:30000"
    inputs = []

    if type == 'text':
        inputs.append(
            {
                "text": data
            }
        )
    else:
        inputs.append(
            {
                "image": data
            })

    payload = {
        "model": "gme-qwen2-vl",
        "input": inputs,
    }

    response = requests.post(url + "/v1/embeddings", json=payload).json()
    emb = [x.get("embedding") for x in response.get("data", [])]

    return emb[0]


def milvus_insert(client, coll_name, partition_name=None, **kwargs):
    """
    向 Milvus 插入数据，支持动态分区
    """
    # 1. 动态分区管理：如果传入了分区名，确保该分区在 Milvus 中已存在
    if partition_name:
        # 检查分区是否存在
        if not client.has_partition(collection_name=coll_name, partition_name=partition_name):
            # 如果不存在，按天自动创建新分区
            client.create_partition(collection_name=coll_name, partition_name=partition_name)
            logger.info(f"成功创建新分区: {partition_name}")

    # 2. 组装实体数据（kwargs 此时不再包含 partition_name，全是纯数据）
    entities = [kwargs]

    # 3. 执行插入操作
    if partition_name:
        # 指定分区插入
        res = client.insert(
            collection_name=coll_name, 
            data=entities, 
            partition_name=partition_name
        )
    else:
        # 退化为默认插入（兼容历史未指定分区的代码）
        res = client.insert(
            collection_name=coll_name, 
            data=entities
        )
        
    return res



def new_insert(client, coll_name, vector, id, savepath, capture_time, device_name):
    entities = [{
        "pic_vector": np.array(vector, dtype=np.float32),
        "path": savepath,
        "capture_time": capture_time,
        "device_name": device_name,
        "id": id

    }]
    res = client.insert(coll_name, entities)
    return res


def vllm2embedding_bge(data, type='img'):
    response = requests.post(
        "http://39.185.65.85:8690/v9v1/embeddings",
        json={
            "model": "BGE-VL-v1.5-zs",
            "messages": [{
                "role": "user",
                "content": [

                    {"type": "image_url", "image_url": {"url": data}},
                    {"type": "text",
                     "text": "重点关注图片中人和车的多标签属性。请注意语义的关联性，例如红色衣服的人、特斯拉品牌的车、手提绿色包包的人"}
                ] if type == "img" else [{"type": "text",
                                          "text": data}],
            }],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()
    return response_json["data"][0]["embedding"]


def sglang2embedding_qwen2_vl(url, base64_qwen,
                              text="找到一张与给定文本匹配的图片。考虑语义关联性，重点关注图片中人、车的多属性特征"):
    payload = {
        "model": "gme-Qwen2-VL-2B-Instruct",
        "input": [
            {
                "text": text
            },
            {
                "image": base64_qwen
            }
        ]
    }

    response = requests.post(url + "/v1/embeddings", json=payload).json()

    return [x.get("embedding") for x in response.get("data", [])]


def base64_to_file(
        base64_str: str,
        file_path: str,
        mode: str = "wb"
) -> bool:
    """
    将Base64字符串转换为文件

    Args:
        base64_str: Base64编码的字符串
        file_path: 输出文件路径
        mode: 文件写入模式（二进制模式推荐"wb"）

    Returns:
        转换成功返回True，失败返回False
    """
    try:
        # 移除可能存在的Base64前缀（如"data:image/jpeg;base64,"）
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]

        # 解码Base64数据
        decoded_data = base64.b64decode(base64_str)

        # 写入文件
        with open(file_path, mode) as f:
            f.write(decoded_data)

        return True
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

def showimg_scale_tools(oriimg, info):
    h, w, c = oriimg.shape
    box_width = info.width()
    box_height = info.height()
    scale = float(os.getenv("SHOW_BOX_SCALE"))
    box_sacel = box_width / box_height
    add_height,add_width = 0,0
    if box_sacel > scale:
        add_height = (box_width / scale) - box_height
    elif box_sacel < scale:
        add_width = (scale - box_sacel) * box_height
    else:
        pass
    xpoints = info.x1
    ypoints = info.y1
    if (xpoints - (add_width/2)) < 0:
        xpoints = 0
    elif (xpoints + box_width + (add_width/2)) > w:
        wabscha = abs(xpoints + box_width + (add_width/2) - w)
        xpoints -= wabscha
    else:
        xpoints -= add_width/2
    if (ypoints - (add_height/2)) < 0:
        ypoints = 0
    elif (ypoints + box_height + (add_height/2)) > h:
        habscha = abs(ypoints + box_height + (add_height/2) - h)
        ypoints -= habscha
    else:
        ypoints -= add_height/2
    box_width +=add_width
    box_height +=add_height
    xpoints,ypoints,box_width,box_height = int(xpoints),int(ypoints),int(box_width),int(box_height)
    cutimg = oriimg[
        ypoints:ypoints+box_height,
        xpoints:xpoints+box_width
    ]
    return cutimg
