# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 10:03:34
# @Author  : 陈澔麟
# @File    : milvus_init.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient
from dotenv import load_dotenv
import os
import re

load_dotenv(".env", override=True)  # override=True确保.env覆盖系统环境变量
client = MilvusClient(
    uri=os.getenv("MILVUS_CLIENT"),
)

# 打印所有字段的名称（注意大小写和符号）
host = None
port = None
# 原始字符串
milvus_client = os.getenv("MILVUS_CLIENT")
# 正则表达式模式，匹配HTTP/HTTPS URL中的host和port
#pattern = r'grpc://([\d.]+):(\d+)'
pattern = r'grpc://(.+):(\d+)'
# 进行匹配
match = re.search(pattern, milvus_client)

if match:
    host = match.group(1)  # 第一个分组是host
    port = match.group(2)  # 第二个分组是port
    print(f"host = {host}")
    print(f"port = {port}")
else:
    print("未匹配到有效的URL格式")

# 检查数据库是否存在
databases = client.list_databases()
db_exists = any(db == os.getenv("MILVUS_DB_NAME") for db in databases)

if not db_exists:
    client.create_database(db_name=os.getenv("MILVUS_DB_NAME"))

# 连接到Milvus服务
connections.connect(
    db_name=os.getenv("MILVUS_DB_NAME"),
    host=host,
    port=port
)


def obj():
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="large_image_url", dtype=DataType.VARCHAR, max_length=255),
        # FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
        FieldSchema(name="device_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="capture_time", dtype=DataType.INT64),
        FieldSchema(name="x1", dtype=DataType.FLOAT),
        FieldSchema(name="x2", dtype=DataType.FLOAT),
        FieldSchema(name="y1", dtype=DataType.FLOAT),
        FieldSchema(name="y2", dtype=DataType.FLOAT),
        FieldSchema(name="target_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="search_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="channel_id", dtype=DataType.INT64),
        FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=580),
        FieldSchema(name="channel_number", dtype=DataType.VARCHAR, max_length=255),  # INT64),
        FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=580)
    ]

    # 集合名称
    collection_name = os.getenv("MILVUS_VECTOR_COLLECTION_NAME")

    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        return

        # utility.drop_collection(collection_name)

    # 创建集合
    schema = CollectionSchema(fields=fields, description="基于IP内积的图片检索集合")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 30}  # 自动选择最优参数
    }

    # 创建索引
    print(f"正在为集合 {collection_name} 创建索引...")
    collection.create_index(
        field_name="vector",
        index_params=index_params,
        index_name="image_vector_index"
    )
    collection.load()

    # 加载集合到内存
    print(f"正在加载集合 {collection_name} 到内存...")
    collection.load()

    # 输出集合信息
    print(f"集合 {collection_name} 创建成功，索引已设置，已加载到内存。")
    print(f"集合描述: {collection.description}")
    print(f"集合字段: {[field.name for field in collection.schema.fields]}")


def obj_filter():
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="large_image_url", dtype=DataType.VARCHAR, max_length=255),
        # FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
        FieldSchema(name="device_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="capture_time", dtype=DataType.INT64),
        FieldSchema(name="x1", dtype=DataType.FLOAT),
        FieldSchema(name="x2", dtype=DataType.FLOAT),
        FieldSchema(name="y1", dtype=DataType.FLOAT),
        FieldSchema(name="y2", dtype=DataType.FLOAT),
        FieldSchema(name="target_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="search_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="channel_id", dtype=DataType.INT64),
        FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=580),
        FieldSchema(name="channel_number", dtype=DataType.VARCHAR, max_length=255),  # INT64),
        FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=580)
    ]

    # 集合名称
    collection_name = os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME")

    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        #return

        utility.drop_collection(collection_name)

    # 创建集合
    schema = CollectionSchema(fields=fields, description="基于IP内积的图片检索集合")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 30}  # 自动选择最优参数
    }

    # 创建索引
    print(f"正在为集合 {collection_name} 创建索引...")
    collection.create_index(
        field_name="vector",
        index_params=index_params,
        index_name="image_vector_index"
    )
    collection.load()

    # 加载集合到内存
    print(f"正在加载集合 {collection_name} 到内存...")
    collection.load()

    # 输出集合信息
    print(f"集合 {collection_name} 创建成功，索引已设置，已加载到内存。")
    print(f"集合描述: {collection.description}")
    print(f"集合字段: {[field.name for field in collection.schema.fields]}")


def face():
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="large_image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="device_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="capture_time", dtype=DataType.INT64),
        FieldSchema(name="x1", dtype=DataType.FLOAT),
        FieldSchema(name="x2", dtype=DataType.FLOAT),
        FieldSchema(name="y1", dtype=DataType.FLOAT),
        FieldSchema(name="y2", dtype=DataType.FLOAT),
        FieldSchema(name="target_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="search_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="channel_id", dtype=DataType.INT64),
        FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=580),
        FieldSchema(name="channel_number", dtype=DataType.VARCHAR, max_length=255),  # INT64),
        FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=580)
    ]

    # 集合名称
    collection_name = os.getenv("MILVUS_FACE_COLLECTION_NAME")

    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        return
        # utility.drop_collection(collection_name)

    # 创建集合
    schema = CollectionSchema(fields=fields, description="基于IP内积的图片检索集合")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 30}  # 自动选择最优参数
    }

    # 创建索引
    print(f"正在为集合 {collection_name} 创建索引...")
    collection.create_index(
        field_name="vector",
        index_params=index_params,
        index_name="face_vector_index"
    )
    collection.load()

    # 加载集合到内存
    print(f"正在加载集合 {collection_name} 到内存...")
    collection.load()

    # 输出集合信息
    print(f"集合 {collection_name} 创建成功，索引已设置，已加载到内存。")
    print(f"集合描述: {collection.description}")
    print(f"集合字段: {[field.name for field in collection.schema.fields]}")


def facedb():
    # 定义字段
    field2 = FieldSchema(name="picture_address", dtype=DataType.VARCHAR, max_length=128)
    field3 = FieldSchema(name="face_info", dtype=DataType.VARCHAR, max_length=128)
    field4 = FieldSchema(name="face_type", dtype=DataType.INT64)
    field5 = FieldSchema(name="face_vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    field6 = FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=128, is_primary=True)
    # 定义集合模式
    schema = CollectionSchema(fields=[field2, field3, field4, field5, field6], description="人脸向量集合")
    collection_name = os.getenv("MILVUS_FACE_DATABASE_COLLECTION_NAME")

    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        return
        # utility.drop_collection(collection_name)

    # 创建集合
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}  # 自动选择最优参数
    }


    # 创建索引
    print(f"正在为集合 {collection_name} 创建索引...")
    collection.create_index(
        field_name="face_vector",
        index_params=index_params,
        index_name="face_vector_index"
    )
    collection.load()

    # 加载集合到内存
    print(f"正在加载集合 {collection_name} 到内存...")
    collection.load()

    # 输出集合信息
    print(f"集合 {collection_name} 创建成功，索引已设置，已加载到内存。")
    print(f"集合描述: {collection.description}")
    print(f"集合字段: {[field.name for field in collection.schema.fields]}")
 


def plate():
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="large_image_url", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
        FieldSchema(name="device_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="capture_time", dtype=DataType.INT64),
        FieldSchema(name="x1", dtype=DataType.FLOAT),
        FieldSchema(name="x2", dtype=DataType.FLOAT),
        FieldSchema(name="y1", dtype=DataType.FLOAT),
        FieldSchema(name="y2", dtype=DataType.FLOAT),
        FieldSchema(name="target_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="search_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="channel_id", dtype=DataType.INT64),
        FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=580),
        FieldSchema(name="channel_number", dtype=DataType.VARCHAR, max_length=255),  # INT64),
        FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=580)
    ]

    # 集合名称
    collection_name = os.getenv("MILVUS_PLATE_COLLECTION_NAME")

    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        return
        # utility.drop_collection(collection_name)

    # 创建集合
    schema = CollectionSchema(fields=fields, description="基于IP内积的图片检索集合")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 30}  # 自动选择最优参数
    }

    # 创建索引
    print(f"正在为集合 {collection_name} 创建索引...")
    collection.create_index(
        field_name="vector",
        index_params=index_params,
        index_name="face_vector_index"
    )
    collection.load()

    # 加载集合到内存
    print(f"正在加载集合 {collection_name} 到内存...")
    collection.load()

    # 输出集合信息
    print(f"集合 {collection_name} 创建成功，索引已设置，已加载到内存。")
    print(f"集合描述: {collection.description}")
    print(f"集合字段: {[field.name for field in collection.schema.fields]}")


if __name__ == '__main__':
    facedb()
    face()
    obj()
    plate()
    obj_filter()

