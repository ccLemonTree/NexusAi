import os

import cv2
import numpy as np
import requests

from api.search.search import research
from tools.init import client
from api.infer.Triton_model.plate.utils.utils import four_point_transform, get_split_merge
from datetime import datetime, timedelta
from api.infer.Triton_model.retina.utils.face_alignment import align_face_and_map_coordinates
import io
from PIL import Image
def cut_img(img,info):
    orih, oriw, oric = img.shape
    ops = info.width() / info.image_width * 300
    text = ""
    if  info.classname =="face":
        objimg = align_face_and_map_coordinates(img, info, target_size=None)
        # objimg = img[
        #          int(np.clip(info.y1 - ops, 0, orih)):int(np.clip(info.y2 + ops, 0, orih)),
        #          int(np.clip(info.x1 - ops, 0, oriw)):int(np.clip(info.x2 + ops, 0, oriw))
        #          ]
    elif info.classname =="PlateSearch-car":
        objimg = img[
                 int(np.clip(info.y1, 0, orih)):int(np.clip(info.y2, 0, orih)),
                 int(np.clip(info.x1, 0, oriw)):int(np.clip(info.x2, 0, oriw))
                 ]
        car_pil = Image.fromarray(cv2.cvtColor(objimg, cv2.COLOR_BGR2RGB))  # 若为 RGB 则改为 Image.fromarray(car)
        # 2. 保存为字节流（用 PNG/JPG 格式均可，根据接口要求调整）
        image_byte_arr = io.BytesIO()
        car_pil.save(image_byte_arr, format='PNG')  # 格式可选 PNG/JPG
        image_byte_arr.seek(0)  # 重置字节流指针到开头
        # 3. 构造 form-data 请求（name 为 "image"）
        lpr_url = os.getenv("LPR_FASTVLM_POST")  # 从环境变量获取接口地址
        files = {
            "image": ("car_crop.png", image_byte_arr, "image/png")  # 依次：文件名、字节流、MIME类型
        }
        response = requests.post(
            url=lpr_url,
            files=files,  # 传入 form-data 数据
            timeout=30  # 超时时间，根据需求调整
        )
        text = response.json()['response']
    else:
        objimg = img[
                 int(np.clip(info.y1 - ops, 0, orih)):int(np.clip(info.y2 + ops, 0, orih)),
                 int(np.clip(info.x1 - ops, 0, oriw)):int(np.clip(info.x2 + ops, 0, oriw))
                 ]
    return text ,objimg

def similarity_if(emb,device_id):

    current_time = datetime.now()
    target_time = current_time - timedelta(seconds=120)
    end_time = int(current_time.timestamp())
    start_time = int(target_time.timestamp())

    filterstrs = f"capture_time > {start_time} and capture_time < {end_time} "
    filterstrs += f"and device_name in {[device_id]}"
    search_data = client.search(
        collection_name=os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME"),
        data=emb,
        anns_field='vector',
        filter=None if filterstrs == "" else filterstrs,
        limit=5,
        output_fields=["id", "desc", "x1", "x2", "y1", "y2", "image_url", "large_image_url", "device_id", "device_name",
                       "channel_id", "channel_name",
                       "channel_number", "capture_time", "target_category"]
    )

    results = []
    for hits in search_data:
        for info in hits:
            results.append( int(info['entity']['capture_time']))
    sim_conf = sum(results) /5
    if sim_conf > float(os.getenv("REVECTOR_SIMTHRESHOLD")):
        return False
    return True