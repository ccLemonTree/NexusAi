import os
import time
from datetime import datetime, timedelta
from tools.logger_tools import CangQiong_Smart_Search_Logger as logger
from tools.api import datetime_to_timestamp
import requests


def tel_rerank(query, search_data):
    rerank_data = requests.post(
        os.getenv("RERANK_TEXT_VECTOR_URL"),
        json={"query": query, "texts": search_data}
    )
    rerank_data.close()
    return rerank_data.json()


def research(client, input, emb, emb_type, target_text=None, filter_divice=None, target=None, start_time=None, end_time=None,
             limit=15):
    collection = "MILVUS_VECTOR_FILTER_COLLECTION_NAME"
    if "人脸" in input:
        collection = "MILVUS_FACE_COLLECTION_NAME"
    elif "车牌" in input:
        collection = "MILVUS_PLATE_COLLECTION_NAME"
    elif "追搜" in input:
        collection = "MILVUS_VECTOR_COLLECTION_NAME"

    filterstrs = ""
    partition_names = []

    if start_time and end_time:
        try:
            dt_start_obj = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            dt_end_obj = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            dt_start_ts = int(dt_start_obj.timestamp())
            dt_end_ts = int(dt_end_obj.timestamp())
            filterstrs = f"capture_time >= {dt_start_ts} and capture_time <= {dt_end_ts}"

            temp_date = dt_start_obj
            while temp_date.date() <= dt_end_obj.date():
                partition_names.append(f"p_{temp_date.strftime('%Y%m%d')}")
                temp_date += timedelta(days=1)

            actual_coll_name = os.getenv(collection)
            existing_partitions = client.list_partitions(collection_name=actual_coll_name)
            partition_names = [p for p in partition_names if p in existing_partitions]

            if not partition_names:
                return []

        except Exception as e:
            logger.error(f"时间解析或分区计算失败: {e}")
            partition_names = None

    if (filter_divice is not None) and isinstance(filter_divice, list) and len(filter_divice) > 0:
        if filterstrs != "":
            filterstrs += f" and device_id in {filter_divice}"
        else:
            filterstrs = f"device_id in {filter_divice}"

    logger.info(f"Milvus查询指令 -> 分区: {partition_names}, 过滤条件: [{filterstrs}]")
    search_data = client.search(
        collection_name=os.getenv(collection),
        data=[emb],
        anns_field='vector',
        partition_names=partition_names,
        filter=None if filterstrs == "" else filterstrs,
        limit=limit,
        output_fields=["id", "desc", "x1", "x2", "y1", "y2", "image_url", "large_image_url",
                       "device_id", "device_name", "channel_id", "channel_name",
                       "channel_number", "capture_time", "target_category"]
    )

    results = []
    for hits in search_data:
        for info in hits:
            results.append({
                'id': info['id'],
                'image_url': info['entity']['image_url'].replace("test/", "search_pic/test/").replace("localhost", "192.168.1.118"),
                'large_image_url': info['entity']['large_image_url'].replace("localhost", "192.168.1.118"),
                'x1': round(info['entity']['x1'], 7),
                'x2': round(info['entity']['x2'], 7),
                'y1': round(info['entity']['y1'], 7),
                'y2': round(info['entity']['y2'], 7),
                'confidence': round(info['distance'], 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(info['entity']['capture_time']))),
                'time': info['entity']['capture_time'],
                'device_id': info['entity']['device_id'],
                'device_name': info['entity']['device_name'],
                'channel_id': info['entity']['channel_id'],
                'channel_name': info['entity']['channel_name'],
                'channel_number': info['entity']['channel_number']
            })
    return results
