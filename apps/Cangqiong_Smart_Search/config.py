import time
import json
import os
import shutil

import httpx
import requests
from fastapi import APIRouter, Request, HTTPException
from pymilvus import Collection
from datetime import datetime, timedelta
from tools.init import cfg, client
router = APIRouter()


# 模型能力查询 （提供当前支持的所有模型）并提供在线状态，不在线需要
@router.post("/objectype")
def MonitoringObjectType():
    message = "succesful"
    data = []
    try:
        data = cfg.nexusDict["Smart_Search"]["basic_label"]
    except Exception as e:
        message = e
    return {
        "message": message,
        "data": data
    }


@router.post("/dataperiod")
def DataStoragePeriod():
    path = os.getenv("OBJ_SAVE_PIC_LOCPATH")
    if os.path.exists(path) == False:
        os.makedirs(path, exist_ok=True)
    # 获取磁盘空间信息
    disk_usage = shutil.disk_usage(path)
    # 转换为GB（1GB = 1024^3字节）
    total_gb = disk_usage.total / (1024 ** 3)
    period = (total_gb - 100) / int(os.getenv("OBJ_MAX_SAVE_GB"))
    return {
        "message": "succesful",
        "data": round(period, 2)
    }


@router.post("/classified")
async def ObjectClassificationStatistics(request: Request):
    request_data = await request.json()

    # 然后从解析后的数据中获取"detectLabels"字段
    detect_labels = request_data.get("detectLabels")
    if len(detect_labels) == 0:
        detect_labels = cfg.nexusDict["Smart_Search"]["basic_label"]
    clsf = {}
    for i in detect_labels:
        result = client.query(
            collection_name=os.getenv("MILVUS_VECTOR_COLLECTION_NAME"),
            filter=f"target_category == \"{i}\"",
            output_fields=["count(*)"]  # 只计算数量
        )
        clsf[i] = result[0]['count(*)']
    return {
        "message": "succesful",
        "data": [clsf]
    }


@router.post("/unitime")
async def unit_recognition_time():
    times = []
    avg_time = 0.3
    url = "http://localhost:10012/ai/seefor/api/search/vlm2vector"
    payload = {"text": "SUV", "limit": 1000}

    # 使用 httpx.AsyncClient 发起异步请求
    async with httpx.AsyncClient() as client:
        for _ in range(10):
            start = time.time()
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()  # 检查 HTTP 错误
            except:
                # 记录错误并继续，避免中断测试
                continue
            finally:
                end = time.time()
                times.append(end - start)

    # 去除最大最小值后求平均
    if len(times) < 3:
        return {"message": "Not enough successful requests", "data": None}

    times.remove(max(times))
    times.remove(min(times))
    avg_time = round(sum(times) / len(times), 2)  # 注意：不一定是 8 次，用实际长度

    return {
        "message": "successful",
        "data": avg_time
    }

def get_today_yesterday():
    """获取当天和昨天的日期字符串（YYYY-MM-DD），适配时区"""
    now = datetime.now() + timedelta(hours=8)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return today, yesterday

@router.get("/vector/count/previous-day")
def get_previous_day_vector_count():
    """
    仅保留今日/昨日count，返回规则：
    1. 有昨日count：返回 今日count - 昨日count
    2. 无昨日count（首次查询）：返回 今日count
    """
    # 1. 校验环境变量
    collection_name = os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME")
    info_json_path = os.getenv("INFO_JSON")
    try:
        # 2. 获取今日/昨日日期 + 当前Milvus总计数
        today, yesterday = get_today_yesterday()
        client.flush(collection_name)
        stats = client.get_collection_stats(collection_name=collection_name)
        today_count = stats.get("row_count", 0)
        # 3. 初始化记录结构（仅保留今日/昨日）
        record_struct = {
            "yesterday_date": yesterday,
            "yesterday_count": 0,
            "today_date": today,
            "today_count": today_count
        }

        # 4. 读取/创建记录文件
        if os.path.isfile(info_json_path):
            try:
                with open(info_json_path, "r", encoding="utf-8") as f:
                    old_record = json.load(f)
            except json.JSONDecodeError:
                old_record = record_struct
        else:
            # 首次查询：无文件，直接初始化并返回今日count
            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(record_struct, f, ensure_ascii=False, indent=2)
            return {"row_count": today_count}

        # 5. 核心逻辑：判断是否跨天，更新昨日count
        if old_record.get("today_date") == today:
            # 未跨天：昨日count沿用旧记录的昨日count
            record_struct["yesterday_count"] = old_record.get("yesterday_count", 0)
        else:
            # 跨天：旧记录的今日count变为新的昨日count
            record_struct["yesterday_count"] = old_record.get("today_count", 0)

        # 6. 计算返回值
        if record_struct["yesterday_count"] == 0:
            # 无昨日count（首次查询/跨天但无历史）：返回今日count
            result_count = today_count
        else:
            # 有昨日count：返回 今日count - 昨日count
            result_count = max(0, today_count - record_struct["yesterday_count"])

        # 7. 写入最新记录（仅保留今日/昨日）
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(record_struct, f, ensure_ascii=False, indent=2)

        # 8. 返回结果
        return {
            "row_count": result_count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/vector/storage-size")
def get_vector_storage_size():
    """获取所有向量的占用磁盘大小（单位：字节）"""
    # 获取集合统计信息
    collection_name = os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME")
    client.flush(collection_name)
    stats = client.get_collection_stats(collection_name=collection_name)
    count = stats.get("row_count", 0)
    vector_bytes = count * 1024 * 4
    # 粗略加上标量字段（假设平均 100 字节/行）
    total_approx = vector_bytes + count * 100
    print(f"🔍 粗略估算占用: {total_approx / (1024 ** 2):.2f} GB")

    return {"row_size": round(total_approx / (1024 ** 2),2)}


@router.get("/vector/count/total")
def get_total_vector_count():
    """获取所有向量的总数"""
    collection_name = os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME")
    client.flush(collection_name)
    stats = client.get_collection_stats(collection_name=collection_name)
    count = stats.get("row_count",0)
    return {"row_count": count}