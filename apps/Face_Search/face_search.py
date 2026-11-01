# -*- coding: utf-8 -*-
# @Time    : 2025/8/29 14:31:06
# @Author  : 陈澔麟
# @File    : face_search.py
import os
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request
from api.vector.vector import gme_vector
from api.search.search import research
from pydantic import BaseModel
from tools.init import client,tritonServer
from tools.logger_tools import CangQiong_Smart_Search_Logger as logger
import cv2
import uuid
import numpy as np
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
    if len(bounding) == 0:
        return None
    objimg = align_face_and_map_coordinates(img, bounding[0], target_size=None)
    bounding = tritonServer.run("dlib_face", objimg, label_to_detect=["face_vector"])[0]
    facevector = np.array(bounding.parames_vector, np.float32)
    return facevector

router = APIRouter()


@router.post('/faceAdd')
async def faceAdd(request: InputFaceData):
    """
    人脸添加
    接口 参数
    param: picture  人脸图片
    param: picType  图片类型
    param: faceType 0 | 1 黑白名单
    param: faceInfo  人员信息，姓名
            {
            人员信息
        }
    return {
        “message”: 200 | "没有检测到人脸"
        “faceid”: 65asd456q21es551
    }

    {
    "picture": ["D:\\Beta_v2\\sample\\img_1.png"],
    "picType": 0,
    "faceType": 0,
    "faceInfo": {
        "name":"刘亦菲",
        "age": 32
    }
}
    """
    picType = request.picType
    faceType = request.faceType
    picture = request.picture
    faceInfo = request.faceInfo
    if picture is None:
        return {'message': 0}
    import urllib.request as brequest
    import numpy as np
    response = brequest.urlopen(picture[0])
    img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    h,w,c = img.shape
    if c>3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    facevector = get_vector(img)
    if facevector is None:
        return JSONResponse(
        content={"message": "未检测到人脸"},
        status_code=400
    )
    uid = uuid.uuid4()
    uuidhex = str(uid).replace('-', '')
    entities = [{
        'uuid': uuidhex,
        'face_type': faceType,
        "face_info": str(faceInfo),
        'picture_address': picture[0],
        'face_vector': facevector
    }]
    res = client.insert(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"), entities)
    logger.info(f"人脸插入成功！{res}")
    client.flush(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"))
    return {'message': 200, "faceId": uuidhex}


# 人脸删除
@router.post('/faceDel')
async def faceDel(request: DelFaceData):
    """
    人脸添加
    接口 参数
    param: faceId  人脸图片

    return {
        “message”: 200 | "没有检测到人脸"
        “id”: 65asd456q21es551
        “vector”：二进制     ？
    }
    删
    """
    faceId = request.faceId
    if faceId is None:
        return {"message": "faceID is None"}
    # 要删除的 id
    target_id = faceId

    # 构造删除表达式
    delete_expr = f'uuid == "{target_id}"'

    try:
        # 执行删除操作
        client.delete(collection_name=os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"), filter=delete_expr)
        print(f"成功删除 id 为 {target_id} 的数据")
        client.flush(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"))
    except Exception as e:
        print(f"删除数据时发生错误: {e}")

    # 刷新集合
    return {'message': 200}


# 人脸更新
@router.post('/faceUpdate')
async def faceUpdate(request: UpdateFaceData):
    """
    人脸添加
    接口 参数
    param: picture  人脸图片
    param: picType  图片类型
    param: faceId 0 | 1 黑白名单
    param: faceInfo  人员信息，姓名
            {
            人员信息
        }
    return {
        “message”: 200 | "没有检测到人脸"
        “id”: 65asd456q21es551
        “vector”：二进制     ？
    }

    {
    "picture": ["D:\\Beta_v2\\sample\\img_1.png","D:\\Beta_v2\\sample\\img_1.png"],
    "picType": 0,
    "faceType": 0,
    "faceInfo": {
        "name":"刘亦菲",
        "age": 32
    }
}
    """
    picType = request.picType
    faceType = request.faceType
    picture = request.picture
    faceInfo = request.faceInfo
    faceId = request.faceId
    if picture is None:
        return {'message': 0}
    # 构造删除表达式
    delete_expr = f'uuid == "{faceId}"'

    try:
        # 执行删除操作
        client.delete(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"), delete_expr)
        logger.info(f"成功删除 id 为 {faceId} 的数据")
    except Exception as e:
        logger.error(f"删除数据时发生错误: {e}")
    import urllib.request as brequest
    import numpy as np
    # response = brequest.urlopen(picture[0])
    # img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    # img = cv2.imdecode(img_array, -1)
    img= cv2.imread(picture[0])
    h, w, c = img.shape
    if c > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    facevector = get_vector(img)
    # facevector = extract_feature(io.imread(picture[0]))
    entities = [{
        'uuid': faceId,
        'face_type': faceType,
        "face_info": str(faceInfo),
        'picture_address': picture[0],
        'face_vector': facevector
    }]
    res = client.insert(os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME"), entities)
    logger.info(f"人脸插入成功！{res}")
    # 刷新集合
    client.flush('face')

    return {'message': 200, "faceId": faceId}                                                          