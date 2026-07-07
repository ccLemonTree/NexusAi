import os
import uuid

import cv2
import numpy as np
import urllib.request as brequest
from fastapi import APIRouter, Request
from pydantic import BaseModel

from api.vector.vector import gme_vector
from api.search.search import research
from tools.init import client, tritonServer
from tools.logger_tools import CangQiong_Smart_Search_Logger as logger
from api.infer.Triton_model.retina.utils.face_alignment import align_face_and_map_coordinates


class InputFaceData(BaseModel):
    picture: list[str] = None
    picType: str = None
    faceType: int = None
    faceInfo: dict = {"name": "无名氏"}


class DelFaceData(BaseModel):
    faceId: str = None


class UpdateFaceData(BaseModel):
    picture: list[str] = None
    picType: str = None
    faceType: int = None
    faceInfo: dict = {"name": "无名氏"}
    faceId: str = None

def get_vector(img):
    bounding = tritonServer.run("retina_face", img, label_to_detect=["face"])
    objimg = align_face_and_map_coordinates(img, bounding[0], target_size=None)
    bounding = tritonServer.run("dlib_face", objimg, label_to_detect=["face_vector"])[0]
    facevector = np.array(bounding.parames_vector, np.float32)
    return facevector

router = APIRouter()


@router.post('/faceAdd')
async def faceAdd(request: InputFaceData):
    picType = request.picType
    faceType = request.faceType
    picture = request.picture
    faceInfo = request.faceInfo
    if picture is None:
        return {'message': 0}
    response = brequest.urlopen(picture[0])
    img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    h, w, c = img.shape
    if c > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    facevector = get_vector(img)
    uid = uuid.uuid4()
    uuidhex = str(uid).replace('-', '')
    entities = [{
        'uuid': uuidhex,
        'face_type': faceType,
        'face_info': str(faceInfo),
        'picture_address': picture[0],
        'face_vector': facevector
    }]
    res = client.insert(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"), entities)
    logger.info(f"人脸插入成功！{res}")
    client.flush(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"))
    return {'message': 200, "faceId": uuidhex}


@router.post('/faceDel')
async def faceDel(request: DelFaceData):
    faceId = request.faceId
    if faceId is None:
        return {"message": "faceID is None"}
    collection_name = os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME")
    delete_expr = f"uuid == '{faceId}'"
    try:
        client.delete(collection_name, delete_expr)
        logger.info(f"成功删除 id 为 {faceId} 的数据")
    except Exception as e:
        logger.error(f"删除数据时发生错误: {e}")
    client.flush(collection_name)
    return {'message': 200}


@router.post('/faceUpdate')
async def faceUpdate(request: UpdateFaceData):
    picType = request.picType
    faceType = request.faceType
    picture = request.picture
    faceInfo = request.faceInfo
    faceId = request.faceId
    if picture is None:
        return {'message': 0}
    collection_name = os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME")
    delete_expr = f"uuid == '{faceId}'"
    try:
        client.delete(collection_name, delete_expr)
        logger.info(f"成功删除 id 为 {faceId} 的数据")
    except Exception as e:
        logger.error(f"删除数据时发生错误: {e}")
    response = brequest.urlopen(picture[0])
    img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    h, w, c = img.shape
    if c > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    facevector = get_vector(img)
    entities = [{
        'uuid': faceId,
        'face_type': faceType,
        'face_info': str(faceInfo),
        'picture_address': picture[0],
        'face_vector': facevector
    }]
    res = client.insert(collection_name, entities)
    logger.info(f"人脸插入成功！{res}")
    client.flush(collection_name)
    return {'message': 200, "faceId": faceId}
