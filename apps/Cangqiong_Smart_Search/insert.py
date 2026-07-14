from datetime import datetime
from fastapi import APIRouter, File, Form, UploadFile
import numpy as np
from PIL import Image
import cv2
import os, requests
from time import time
from typing import List, Optional
import base64
from api.vector.vector import gme_vector
from api.search.search import research
from tools.api import milvus_insert, showimg_scale_tools
from tools.init import tritonServer
from pydantic import BaseModel
from tools.init import client, chat_infer
from api.infer.utils import siou
from tools.logger_tools import CangQiong_Smart_Insert_Logger as logger
import asyncio
from apps.Cangqiong_Smart_Analyse.analyse import executor
from api.infer.Utils.result_utils import Unlogic_run
from api.infer.Utils.analyse_utils import LabelToModel
from api.infer.running import analyseRun
from apps.Cangqiong_Smart_Search.utils.utils import cut_img, similarity_if
from tools.init import cfg
import uuid
from io import BytesIO
import re

router = APIRouter()
milvus_type = {
    "face": "MILVUS_FACE_COLLECTION_NAME",
    "PlateSearch-car": "MILVUS_PLATE_COLLECTION_NAME"
}

pixelmax_label = {
    "face": 1000,
    "PlateSearch-car": 1000,
}


class InputRequestData(BaseModel):
    capture_time: int
    device_name: str
    device_id: str
    pic_path: str
    pic_url: str = ""
    channel_id: int
    channel_name: str
    channel_number: str

def read_image(input_data):
    base64_pattern = r'^data:image/[a-zA-Z0-9]+;base64,'
    if isinstance(input_data, str):
        if re.match(base64_pattern, input_data):
            try:
                base64_data = re.sub(base64_pattern, '', input_data)
                image_bytes = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_bytes))
            except (base64.binascii.Error, IOError) as e:
                raise ValueError(f"Base64图片解析失败：{str(e)}")

        try:
            image_bytes = base64.b64decode(input_data, validate=True)
            return Image.open(BytesIO(image_bytes))
        except base64.binascii.Error:
            pass

    if isinstance(input_data, str) and input_data.startswith(('http://', 'https://')):
        try:
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"网络图片解析失败：{str(e)}")

    try:
        return Image.open(input_data)
    except (IOError, FileNotFoundError) as e:
        raise ValueError(f"本地图片读取失败：{str(e)}")


@router.post("/vec2milvus")
async def vec2milvus(request: InputRequestData):
    try:
        now = datetime.fromtimestamp(int(request.capture_time))
    except Exception as e:
        logger.error(f"时间戳解析失败，使用当前时间兜底: {e}")
        now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')

    partition_name = f"p_{now.strftime('%Y%m%d')}"

    setsLabel = cfg.nexusDict["Smart_Search"]["basic_label"]
    logger.info(f"from [{request.device_id}] --dict input= ")
    obj_root = os.getenv("OBJ_SAVE_PIC_LOCPATH")
    if not obj_root:
        return {"message": "未配置 OBJ_SAVE_PIC_LOCPATH"}
    try:
        pil_image = read_image(request.pic_path)
        img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.info(f"from [{request.device_id}] --error  {e}")
        return {"message": "插入失败"}
    firstAnalysis = setsLabel
    loop = asyncio.get_event_loop()
    boundings = await loop.run_in_executor(executor, analyseRun, firstAnalysis, [img])
    logger.info(f"from [{request.device_id}] --inputs：{firstAnalysis} -==:{boundings}")
    logger.info(f"from [{request.device_id}] --boundings：{boundings}")
    base_dir = os.path.join(obj_root, request.device_id, year, month, day)
    if request.pic_url:
        large_image_url = request.pic_url
        if len(boundings) == 0:
            logger.info(f"from [{request.device_id}] --analyse：[]")
        else:
            logger.info(f"from [{request.device_id}] --analyse_obj：{boundings}]")
            
    else:
        large_image_url = os.path.join(base_dir, f"{uuid.uuid4()}.jpeg")
        if len(boundings) == 0:
            logger.info(f"from [{request.device_id}] --analyse：[]")
        else:
            os.makedirs(base_dir, exist_ok=True)
            pil_image.save(large_image_url)
            logger.info(f"from [{request.device_id}] --analyse_obj：{boundings}]")
    jsq = 0
    for k, info in enumerate(boundings):
        text = ""
        _, objimg = cut_img(img, info)
        h, w, c = objimg.shape
        if (h * w) < pixelmax_label.get(info.classname, 20000):
            continue

        vector, emb_type = await gme_vector(
            question=text,
            pic_path=objimg,
            prompt=os.getenv("SYSTEM_PROMPT"),
            insert_type=info.classname
        )

        try:
            milvus_insert(
                client,
                coll_name=os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME"),
                partition_name=partition_name,
                desc=text,
                image_url=large_image_url,
                large_image_url=large_image_url,
                vector=vector,
                device_id=request.device_id,
                capture_time=request.capture_time,
                device_name=request.device_name,
                channel_id=request.channel_id,
                channel_name=request.channel_name,
                channel_number=request.channel_number,
                x1=info.u1,
                x2=info.u2,
                y1=info.v1,
                y2=info.v2,
                target_category=info.classname,
                search_type="obj"
            )
            jsq += 1
        except Exception as e:
            logger.info(f"插入失败{e}")
        finally:
            logger.info(f"from [{request.device_id}] -- end")
    logger.info(f"from [{request.device_id}] 保存向量个数 {jsq}")
    return {"message": "插入成功", "data": jsq}


class InputTestRequestData(BaseModel):
    capture_time: int = 1762407805
    device_name: str = "高德"
    device_id: str = "AVCW002W"
    pic_path: str
    channel_id: int = 1111111111
    channel_name: str = "111"
    channel_number: str = "w222"


@router.post("/testvec2milvus")
async def test_vec2milvus(request: InputTestRequestData):
    pil_image = Image.open(request.pic_path)
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    vector, emb_type = await gme_vector(
        question="",
        pic_path=request.pic_path,
        prompt=os.getenv("SYSTEM_PROMPT"),
        insert_type=False
    )
    try:
        milvus_insert(
            client,
            coll_name=os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME"),
            desc="text",
            image_url=r"http://localhost:10012/image/" + request.pic_path,
            large_image_url=r"http://localhost:10012/image/" + request.pic_path,
            vector=np.array(vector, dtype=np.float32),
            device_id=request.device_id,
            capture_time=request.capture_time,
            device_name=request.device_name,
            channel_id=request.channel_id,
            channel_name=request.channel_name,
            channel_number=request.channel_number,
            x1=0.0,
            x2=0.0,
            y1=0.0,
            y2=0.0,
            target_category="tt",
            search_type="obj"
        )
    except Exception as e:
        return {"message": f"插入失败{e}"}

    return {"message": "插入成功"}


@router.post("/vec2milvusbyte")
async def vec2milvus_by_file(
    deviceId: str = Form(default=""),
    captureTime: str = Form(default=""),
    deviceName: str = Form(default=""),
    channelId: str = Form(default=""),
    channelName: str = Form(default=""),
    channelNumber: str = Form(default=""),
    picUrl: str = Form(default=""),
    file: UploadFile = File(...)
):
    try:
        ts = int(captureTime)
        if ts > 10000000000:
            ts = ts // 1000
        now = datetime.fromtimestamp(ts)
    except Exception as e:
        logger.error(f"时间戳解析失败，使用当前时间兜底: {e}")
        now = datetime.now()
        ts = int(now.timestamp())
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    partition_name = f"p_{now.strftime('%Y%m%d')}"

    setsLabel = cfg.nexusDict["Smart_Search"]["basic_label"]

    obj_root = os.getenv("OBJ_SAVE_PIC_LOCPATH")
    if not obj_root:
        return {"message": "未配置 OBJ_SAVE_PIC_LOCPATH"}

    try:
        import io
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content))
        img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.info(f"from [{deviceId}] --error {e}")
        return {"message": "插入失败"}

    loop = asyncio.get_event_loop()
    boundings = await loop.run_in_executor(executor, analyseRun, setsLabel, [img])

    base_dir = os.path.join(obj_root, deviceId, year, month, day)
    if picUrl:
        large_image_url = picUrl
        if len(boundings) == 0:
            logger.info(f"from [{deviceId}] --analyse：[]")
        else:
            logger.info(f"from [{deviceId}] --analyse_obj：{boundings}]")
            
    else:
        large_image_url = os.path.join(base_dir, f"{uuid.uuid4()}.jpeg")
        if len(boundings) == 0:
            logger.info(f"from [{deviceId}] --analyse：[]")
        else:
            os.makedirs(base_dir, exist_ok=True)
            pil_image.save(large_image_url)
            logger.info(f"from [{deviceId}] --analyse_obj：{boundings}]")


    jsq = 0
    for info in boundings:
        text = ""
        _, objimg = cut_img(img, info)

        h, w, c = objimg.shape
        if (h * w) < pixelmax_label.get(info.classname, 20000):
            continue

        vector, emb_type = await gme_vector(
            question=text,
            pic_path=objimg,
            prompt=os.getenv("SYSTEM_PROMPT"),
            insert_type=info.classname
        )

        try:
            milvus_insert(
                client,
                coll_name=os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME"),
                partition_name=partition_name,
                desc=text,
                image_url=large_image_url,
                large_image_url=large_image_url,
                vector=vector,
                device_id=deviceId,
                capture_time=ts,
                device_name=deviceName,
                channel_id=int(channelId) if channelId else 0,
                channel_name=channelName,
                channel_number=channelNumber,
                x1=info.u1,
                x2=info.u2,
                y1=info.v1,
                y2=info.v2,
                target_category=info.classname,
                search_type="obj"
            )
            jsq += 1
        except Exception as e:
            logger.info(f"插入失败{e}")

    logger.info(f"from [{deviceId}] 保存向量个数 {jsq}")
    return {"message": "插入成功", "data": jsq}
