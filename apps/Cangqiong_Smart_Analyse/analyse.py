import asyncio
import copy
from fastapi import APIRouter, Form, File, UploadFile
import numpy as np
import cv2
import os
import uuid
import json
import traceback
import httpx

from api.infer.Utils.utils import imgRead
from api.infer.Utils.eos import load_eos_image
from tools.init import client, chat_infer, cfg
from api.infer.running import analyseRun
from tools.init import tritonServer
from api.infer.Utils.result_utils import logic_run
from api.infer.Utils.class_info import CameraInfo
from tools.concurrency import get_executor, get_logic_executor
from tools.logger_tools import CangQiong_Smart_Model_logger as logger_model
from tools.logger_tools import CangQiong_Smart_Vllm_logger as logger_vllm
from fastapi.responses import JSONResponse
import time
import base64
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


request_executor = get_executor(
    "REQUEST_MAX_WORKERS", default_scale=1, hard_cap=8, prefix="request"
)
logic_executor = get_logic_executor()


router = APIRouter()


async def _read_analyse_images(camerInfo, error_details=None):
    if camerInfo.picType == "eos":
        kwargs = {"error_details": error_details} if error_details is not None else {}
        images = await asyncio.gather(*(load_eos_image(key, **kwargs) for key in camerInfo.imgsList))
        return [image for image in images if image is not None]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(request_executor, imgRead, camerInfo)


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
    deviceId: str = ""
    presetId: str = ""
    basename_pic: str = ""
    labelDetails: List[dict]


# bbox子模型
class ImageBboxItem(BaseModel):
    label: str
    confidence: float
    ltx: int
    lty: int
    width: int
    height: int
    otherInfo: Dict[str, Any]
    ImageWidth: int
    ImageHeight: int

# 图片列表子模型
class ImageListItem(BaseModel):
    data_type: str = Field(description="枚举: eos | url | base64 | localfile")
    data_value: str
    image_bbox: List[ImageBboxItem] = Field(default_factory=list)

# 请求模型，对齐之前设计的json字段
class VLLMFrameRequest(BaseModel):
    prompt: str
    question: str
    image_list: List[ImageListItem]


# 返回结构子模型
class FrameResultItem(BaseModel):
    label: str
    isAlarm: bool
    desc: str

# 返回模型
class VLLMFrameResponse(BaseModel):
    message: str
    result: List[FrameResultItem]
    timeout: float




@router.post("/utils/vllm/frame", response_model=VLLMFrameResponse)
async def openai_vllm_frame(request: VLLMFrameRequest):
    message = "successful"
    start_total = time.time()
    result_items = []

    try:
        system = request.prompt
        question = request.question
        image_list = request.image_list

        if not image_list:
            raise ValueError("image_list 不能为空")

        # === 1. 解码所有图片（EOS 并发下载） ===
        # 1a. 收集非EOS图片直接解码，EOS图片并发下载
        eos_tasks = {}  # {idx: img_item}
        direct_imgs = {}  # {idx: decoded_img}

        for idx, img_item in enumerate(image_list):
            data_type = img_item.data_type
            data_value = img_item.data_value

            try:
                if data_type == "eos":
                    # EOS 图片收集起来并发下载
                    eos_tasks[idx] = img_item
                elif data_type == "url":
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.get(data_value)
                        resp.raise_for_status()
                        img_bytes = resp.content
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        direct_imgs[idx] = img
                    else:
                        logger_vllm.warning(f"第{idx}张 URL 图片解码失败")
                elif data_type == "base64":
                    img_bytes = base64.b64decode(data_value)
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        direct_imgs[idx] = img
                    else:
                        logger_vllm.warning(f"第{idx}张 base64 图片解码失败")
                elif data_type == "localfile":
                    with open(data_value, "rb") as f:
                        img_bytes = f.read()
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        direct_imgs[idx] = img
                    else:
                        logger_vllm.warning(f"第{idx}张本地文件图片解码失败")
                else:
                    logger_vllm.warning(f"未知的 data_type 类型: {data_type}, 跳过第{idx}张图片")
            except Exception as e:
                logger_vllm.warning(f"第{idx}张图片({data_type})处理异常: {e}")

        # 1b. 并发下载所有 EOS 图片
        eos_results = {}
        eos_errors = {}
        if eos_tasks:
            logger_vllm.info(f"开始并发下载 {len(eos_tasks)} 张 EOS 图片")

            async def _download_eos_idx(idx, img_item):
                errors = []
                try:
                    logger_vllm.info(f"第{idx}张图片 EOS 下载: key={img_item.data_value}")
                    img = await load_eos_image(img_item.data_value, error_details=errors)
                    return (idx, img, errors)
                except Exception as e:
                    errors.append(f"key={img_item.data_value}: [{type(e).__name__}] {e}")
                    return (idx, None, errors)

            coros = [_download_eos_idx(idx, item) for idx, item in eos_tasks.items()]
            eos_outcomes = await asyncio.gather(*coros, return_exceptions=True)

            for outcome in eos_outcomes:
                if isinstance(outcome, Exception):
                    logger_vllm.error(f"EOS 并发下载异常: {outcome}")
                else:
                    idx, img, errors = outcome
                    if img is not None:
                        eos_results[idx] = img
                    else:
                        detail = "; ".join(errors)
                        eos_errors[str(idx)] = detail
                        logger_vllm.warning(f"第{idx}张 EOS 图片下载失败: {detail}")

        # 1c. 按顺序合并所有解码结果
        decoded_imgs = []
        for idx in range(len(image_list)):
            if idx in direct_imgs:
                decoded_imgs.append(direct_imgs[idx])
            elif idx in eos_results:
                decoded_imgs.append(eos_results[idx])

        failed_indices = [
            str(idx) for idx in range(len(image_list))
            if idx not in direct_imgs and idx not in eos_results
        ]
        if failed_indices:
            details = "; ".join(f"第{idx}张: {eos_errors[idx]}" if idx in eos_errors else idx for idx in failed_indices)
            raise ValueError(f"图片下载或解码失败: {details}")

        if not decoded_imgs:
            raise ValueError("所有图片均解码失败")

        # === 2. 从第一张图片提取 bbox，对所有图片裁剪 ===
        first_bbox_list = image_list[0].image_bbox
        if not first_bbox_list:
            raise ValueError("第一张图片缺少 image_bbox 标注信息")

        # 使用第一个 bbox 作为裁剪区域
        bbox = first_bbox_list[0]
        # 获取第一个 box 的 label 名称
        bbox_label = bbox.label

        # 裁剪时扩展一点边界
        pad = 90
        x1 = max(0, int(bbox.ltx) - pad)
        y1 = max(0, int(bbox.lty) - pad)
        x2 = int(bbox.ltx) + int(bbox.width) + pad
        y2 = int(bbox.lty) + int(bbox.height) + pad

        # 裁剪所有图片
        cropped_imgs = []


        for idx, img in enumerate(decoded_imgs):
            ih, iw = img.shape[:2]
            cx1 = max(0, min(x1, iw - 1))
            cy1 = max(0, min(y1, ih - 1))
            cx2 = max(0, min(x2, iw))
            cy2 = max(0, min(y2, ih))
            if cx2 > cx1 and cy2 > cy1:
                crop = img[cy1:cy2, cx1:cx2]
                cropped_imgs.append(crop)
            else:
                logger_vllm.warning(f"第{idx}张图片裁剪区域无效，已跳过")

        if not cropped_imgs:
            raise ValueError("所有图片裁剪后均为空")

        logger_vllm.info(
            f"frame解码完成: 原始{len(image_list)}张, 解码{len(decoded_imgs)}张, 裁剪{len(cropped_imgs)}张, "
            f"bbox=[{x1},{y1},{x2},{y2}], label={bbox_label}"
        )

        # === 3. 调用 VLLM 推理 ===
        loop = asyncio.get_running_loop()
        raw_result = await loop.run_in_executor(
            request_executor, chat_infer._frame_infer, system, question, cropped_imgs
        )
        logger_vllm.info(f"frame raw_result type: {type(raw_result)}, value: {raw_result}")

        # === 4. 封装结果 ===
        # raw_result 格式: [{"desc": "xxx", "has_fire_smoke": True/False}]
        is_alarm = False
        desc = ""

        if isinstance(raw_result, list) and len(raw_result) > 0:
            first_item = raw_result[0]
            if isinstance(first_item, dict):
                is_alarm = first_item.get("has_fire_smoke", False)
                desc = first_item.get("desc", "")
            else:
                desc = str(first_item)
                is_alarm = bool(raw_result)
        elif isinstance(raw_result, dict):
            is_alarm = raw_result.get("has_fire_smoke", False)
            desc = raw_result.get("desc", "") or raw_result.get("reason", "") or str(raw_result)
        elif raw_result:
            is_alarm = True
            desc = str(raw_result)
        if is_alarm and os.getenv("SAVECROP")=="1":
            path = os.path.join(f"/ai/capture/aichen/{str(uuid.uuid4())}")
            os.makedirs(path, exist_ok=True)
            for idx,crop in enumerate(cropped_imgs):
                cv2.imwrite(f"{path}/{idx}.png", crop)

        result_items = [{
            "label": bbox_label,
            "isAlarm": bool(is_alarm),
            "desc": desc
        }]
        logger_vllm.info(f"frame result_items: {result_items}")

    except Exception as e:
        message = str(e)
        logger_vllm.error(f"frame接口异常: {traceback.format_exc()}")

    timeout_total = time.time() - start_total

    response_data = {
        "message": message,
        "result": result_items,
        "timeout": round(timeout_total, 4)
    }

    label_val = result_items[0]['label'] if result_items else 'N/A'
    is_alarm_val = result_items[0]['isAlarm'] if result_items else 'N/A'
    logger_vllm.info(
        f"frame接口 | message={message} | timeout={response_data['timeout']}s | "
        f"label={label_val} | isAlarm={is_alarm_val}"
    )

    return response_data

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
    first_stage_rules = {}

    for lab in setsLabel:
        rule = label_rules.get(lab)
        first_stage_labels = cfg.logicModelDict[lab][0]["label"] if lab in logicLabels else [lab]
        if rule:
            for first_stage_label in first_stage_labels:
                current_rule = first_stage_rules.get(first_stage_label)
                if current_rule is None or rule.get("conf", 0) < current_rule.get("conf", 0):
                    first_stage_rules[first_stage_label] = rule
        if lab in logicLabels:
            for i in cfg.logicModelDict[lab][0]["label"]:
                logicAnalysis.add(i)
        else:
            unlogicAnalysis.add(lab)

    firstAnalysis = unlogicAnalysis | logicAnalysis
    firstResult = analyseRun(firstAnalysis, camerInfo.imgsList, camerInfo, first_stage_rules)

    fuctureList = []
    logicAnalysisDict = {}
    unlogicAnalysisList = []

    for lab in setsLabel:
        logicFlag = lab in logicLabels
        comparisonList = cfg.logicModelDict[lab][0]["label"] if logicFlag else [lab]
        conf_thres = label_rules.get(lab, {}).get("conf", 0)
        firstResultInLogic = [
            b for b in firstResult
            if ((b.classname in comparisonList) or (b.classname in logicLabels))
            and b.confidence >= conf_thres
        ]
        if logicFlag:
            logicResultList = logicAnalysisDict.get(lab, [])
            logicResultList.extend(firstResultInLogic)
            logicAnalysisDict[lab] = logicResultList
        else:
            unlogicAnalysisList.extend(firstResultInLogic)

    for key, boundingboxs in logicAnalysisDict.items():
        fuctureList.append(logic_executor.submit(
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
    request_data = request.dict(exclude={"inferData"})
    request_data["inferData_count"] = len(request.inferData)
    logger_model.info("/run/v1 request: %s", request_data)
    camerInfo = CameraInfo()
    camerInfo.deviceId = request.deviceId
    camerInfo.presetId = request.presetId
    camerInfo.picType = request.location
    camerInfo.imgsList = request.inferData
    camerInfo.baseName = request.basename_pic
    label_rules = {j['label']: j['analysisParams'] for j in request.labelDetails}
    logger_model.info(f"DeviceID [{camerInfo.deviceId}] | {label_rules}")

    eos_errors = []
    camerInfo.imgsList = await _read_analyse_images(camerInfo, error_details=eos_errors)
    if not camerInfo.imgsList:
        is_eos = camerInfo.picType == "eos"
        error_detail = "; ".join(eos_errors) if eos_errors else "未提供 EOS 图片 key。" if is_eos else "请检查图片数据是否完整、是否为合法图片格式。"
        logger_model.warning(f"DeviceID [{camerInfo.deviceId}] -- {error_detail}")
        return JSONResponse(
            status_code=400,
            content={
                "message": "图片下载失败" if is_eos else "图片解码失败",
                "error_detail": error_detail
            }
        )

    setsLabel = set(request.inferLabels)
    loop = asyncio.get_running_loop()
    allAnalysisDict = await loop.run_in_executor(
        request_executor, _run_analyse_sync, camerInfo, setsLabel, label_rules)

    logger_model.info(allAnalysisDict)
    return allAnalysisDict


@router.post("/run/v1byte")
async def analyse_byte(
    deviceId: str = Form(default=""),
    presetId: str = Form(default=""),
    inferLabels: str = Form(...),
    labelDetails: str = Form(...),
    location: str = Form(default="file"),
    inferData: str = Form(default=""),
    files: Optional[List[UploadFile]] = File(default=None, description="需要推理的图片")
):
    try:
        logger_model.info(f"labelDetails:------{labelDetails}")
        logger_model.info(f"inferLabels:------{inferLabels}")
        start_total = time.time()

        label_rules = {j['label']: j['analysisParams'] for j in json.loads(labelDetails)}
        logger_model.info(f"label_rules:------{label_rules}")

        camerInfo = CameraInfo()
        camerInfo.deviceId = deviceId
        camerInfo.presetId = presetId
        camerInfo.picType = location
        loop = asyncio.get_running_loop()

        if location == "eos":
            eos_keys = json.loads(inferData)
            if not isinstance(eos_keys, list):
                raise ValueError("inferData 必须是 EOS key 的 JSON 数组")
            camerInfo.imgsList = eos_keys
            imgs = await _read_analyse_images(camerInfo)
        else:
            if not files:
                raise ValueError("需要上传至少一个图片文件")
            raw_contents = [await f.read() for f in files]

            def decode_images(contents):
                imgs = []
                for content in contents:
                    nparr = np.frombuffer(content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        imgs.append(img)
                return imgs

            imgs = await loop.run_in_executor(request_executor, decode_images, raw_contents)
        if not imgs:
            logger_model.warning(f"DeviceID [{deviceId}] -- 图片解码失败")
            return JSONResponse(
                status_code=400,
                content={
                    "message": "图片解码失败",
                    "error_detail": "请检查上传的文件流是否完整、是否为合法图片格式。"
                }
            )
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
            request_executor, _run_analyse_sync, camerInfo, setsLabel, label_rules)

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
