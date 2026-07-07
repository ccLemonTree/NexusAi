import asyncio
import copy
from datetime import datetime
from fastapi import APIRouter, Form, File, UploadFile
import numpy as np
from PIL import Image
import cv2
import os
import time
import base64
import uuid
from time import time as time_func
from typing import List, Optional
import json
import traceback

from api.infer.Utils.utils import imgRead
from pydantic import BaseModel
from tools.init import client, chat_infer, cfg
from api.infer.running import analyseRun
from tools.init import tritonServer

from api.infer.Utils.result_utils import logic_run
from api.infer.Utils.class_info import CameraInfo
from tools.concurrency import get_logic_executor
from tools.logger_tools import CangQiong_Smart_Model_logger as logger_model
from tools.logger_tools import CangQiong_Smart_Vllm_logger as logger_vllm
from fastapi.responses import JSONResponse

executor = get_logic_executor()
router = APIRouter()


class VLLMRequestModel(BaseModel):
    system: str
    question: str
    inferData: List[str]
    location: str
    scale: Optional[float] = 1.0


class AnalyseRequestModel(BaseModel):
    location: str
    inferLabels: List[str]
    inferData: List[str]
    device_id: str = ""
    preset_id: str = ""
    basename_pic: str = ""
    labelDetails: List[dict]


_current_sec = int(time.time())
_req_count = 0

@router.post("/utils/vllm")
async def openai_vllm(request: VLLMRequestModel):
    global _current_sec, _req_count

    now_sec = int(time.time())
    if now_sec == _current_sec:
        _req_count += 1
    else:
        logger_vllm.info(f"[QPS Monitor] 当前 Worker 在过去 1 秒内接收了 {_req_count} 个请求")
        _current_sec = now_sec
        _req_count = 1

    message = "succesful"
    start_total = time.time()
    result = []
    t_decode, t_infer = 0.0, 0.0

    try:
        system = request.system
        question = request.question
        inferData = request.inferData
        location = request.location

        t0 = time.time()
        if location == "base64":
            img_bytes = base64.b64decode(inferData[0])
        elif location == "file":
            with open(inferData[0], "rb") as f:
                img_bytes = f.read()
        else:
            raise ValueError(f"未知的 location 类型: {location}")

        t_decode = time.time() - t0

        t1 = time.time()
        result = chat_infer.infer(system, question, file=img_bytes)
        t_infer = time.time() - t1

    except Exception as e:
        message = str(e)

    timeout = time.time() - start_total

    logger_vllm.info(
        f"[极速链路耗时] 接口总耗时: {timeout:.4f}s | "
        f"纯内存解码: {t_decode:.4f}s | "
        f"异步推理(含网络): {t_infer:.4f}s"
    )
    logger_vllm.info(f"message: {message} \nquestion: {question} \nresult: {result}")

    return {"message": message, "result": [result], "timeout": timeout}


@router.post("/utils/vllmbyte")
async def openai_vllm_byte(
    system: str = Form(default=""),
    question: str = Form(...),
    file: UploadFile = File(...)
):
    global _current_sec, _req_count

    now_sec = int(time.time())
    if now_sec == _current_sec:
        _req_count += 1
    else:
        logger_vllm.info(f"[QPS Monitor] 当前 Worker 在过去 1 秒内接收了 {_req_count} 个请求")
        _current_sec = now_sec
        _req_count = 1

    message = "succesful"
    start_total = time.time()
    result = []
    t_decode, t_infer = 0.0, 0.0

    try:
        t0 = time.time()
        img_bytes = await file.read()
        t_decode = time.time() - t0

        t1 = time.time()
        result = chat_infer.infer(system, question, file=img_bytes)
        t_infer = time.time() - t1

    except Exception as e:
        message = str(e)

    timeout = time.time() - start_total

    logger_vllm.info(
        f"[极速链路耗时] 接口总耗时: {timeout:.4f}s | "
        f"纯内存读取: {t_decode:.4f}s | "
        f"异步推理(含网络): {t_infer:.4f}s"
    )
    logger_vllm.info(f"message: {message} \nquestion: {question} \nresult: {result}")

    return {
        "message": message,
        "result": [result] if result else [],
        "timeout": timeout
    }


def _run_analyse_sync(camerInfo, setsLabel, label_rules):
    logicLabels = list(cfg.logicModelDict.keys())
    unlogicAnalysis = set()
    logicAnalysis = set()

    for lab in setsLabel:
        if lab in logicLabels:
            for i in cfg.logicModelDict[lab][0]["label"]:
                logicAnalysis.add(i)
        else:
            unlogicAnalysis.add(lab)

    firstAnalysis = unlogicAnalysis | logicAnalysis
    firstResult = analyseRun(firstAnalysis, camerInfo.imgsList, camerInfo, label_rules)

    fuctureList = []
    logicAnalysisDict = {}
    unlogicAnalysisList = []

    for lab in setsLabel:
        logicFlag = lab in logicLabels
        comparisonList = cfg.logicModelDict[lab][0]["label"] if logicFlag else [lab]
        firstResultInLogic = [
            b for b in firstResult
            if (b.classname in comparisonList) or (b.classname in logicLabels)
        ]
        if logicFlag:
            logicResultList = logicAnalysisDict.get(lab, [])
            logicResultList.extend(firstResultInLogic)
            logicAnalysisDict[lab] = logicResultList
        else:
            unlogicAnalysisList.extend(firstResultInLogic)

    for key, boundingboxs in logicAnalysisDict.items():
        fuctureList.append(executor.submit(
            logic_run, camerInfo.imgsList, copy.deepcopy(boundingboxs), camerInfo, key, tritonServer, label_rules))

    logicResults = []
    for fucture in fuctureList:
        result = fucture.result()
        if len(list(result.values())[0]['bbox']) == 0:
            continue
        for key, value in result.items():
            for i in range(len(value['bbox'])):
                value['bbox'][i] = value['bbox'][i].dict()
        logicResults.append(result)

    unlogicAnalysisDict = {}
    for info in unlogicAnalysisList:
        if info.classname in setsLabel:
            get_unlogic_result = unlogicAnalysisDict.get(info.classname, [])
            get_unlogic_result.append(info.dict())
            unlogicAnalysisDict[info.classname] = get_unlogic_result

    unlogicAnalysisDicts = {k: {'bbox': v, "imglist": []} for k, v in unlogicAnalysisDict.items()}
    for i in logicResults:
        unlogicAnalysisDicts.update(i)

    return unlogicAnalysisDicts


@router.post("/run/v1")
async def analyse(request: AnalyseRequestModel):
    camerInfo = CameraInfo()
    camerInfo.deviceId = request.device_id
    camerInfo.presetId = request.preset_id
    camerInfo.picType = request.location
    camerInfo.imgsList = request.inferData
    camerInfo.baseName = request.basename_pic
    label_rules = {j['label']: j['analysisParams'] for j in request.labelDetails}
    logger_model.info(f"DeviceID [{camerInfo.deviceId}] | {label_rules}")

    loop = asyncio.get_event_loop()
    camerInfo.imgsList = await loop.run_in_executor(executor, imgRead, camerInfo)

    setsLabel = set(request.inferLabels)
    allAnalysisDict = await loop.run_in_executor(
        executor, _run_analyse_sync, camerInfo, setsLabel, label_rules)

    logger_model.info(allAnalysisDict)
    return allAnalysisDict


@router.post("/run/v1byte")
async def analyse_byte(
    deviceId: str = Form(default=""),
    presetId: str = Form(default=""),
    inferLabels: str = Form(...),
    labelDetails: str = Form(...),
    files: List[UploadFile] = File(..., description="需要推理的图片")
):
    try:
        logger_model.info(f"labelDetails:------{labelDetails}")
        logger_model.info(f"inferLabels:------{inferLabels}")
        start_total = time.time()

        label_rules = {j['label']: j['analysisParams'] for j in json.loads(labelDetails)}
        logger_model.info(f"label_rules:------{label_rules}")

        raw_contents = [await f.read() for f in files]

        def decode_images(contents):
            imgs = []
            for content in contents:
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
            return imgs

        loop = asyncio.get_event_loop()
        imgs = await loop.run_in_executor(executor, decode_images, raw_contents)
        if not imgs:
            logger_model.warning(f"DeviceID [{deviceId}] -- 图片解码失败")
            return JSONResponse(
                status_code=400,
                content={
                    "message": "图片解码失败",
                    "error_detail": "请检查上传的文件流是否完整、是否为合法图片格式。"
                }
            )
        camerInfo = CameraInfo()
        camerInfo.deviceId = deviceId
        camerInfo.presetId = presetId
        camerInfo.imgsList = imgs

        logger_model.info(f"DeviceID [{camerInfo.deviceId}] | {label_rules}")

        try:
            actual_labels = json.loads(inferLabels)
            if isinstance(actual_labels, list):
                setsLabel = set(actual_labels)
            else:
                setsLabel = {str(actual_labels)}
        except Exception:
            setsLabel = {inferLabels}

        allAnalysisDict = await loop.run_in_executor(
            executor, _run_analyse_sync, camerInfo, setsLabel, label_rules)

        timeout = time.time() - start_total
        logger_model.info(f"小模型推理总耗时: {timeout:.4f}s ")
        logger_model.info(f"deviceId:{deviceId}小模型返回结果：{allAnalysisDict}")

        return allAnalysisDict

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger_model.error(
            f"======== 严重预警：接口内部发生异常 ========\n"
            f"DeviceID: [{deviceId}]\n"
            f"错误信息: {str(e)}\n"
            f"详细堆栈:\n{error_traceback}\n"
            f"============================================="
        )
        return JSONResponse(
            status_code=500,
            content={
                "message": "Python backend internal error",
                "error_detail": str(e)
            }
        )
