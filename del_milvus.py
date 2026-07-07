import os
import re
import shutil
from datetime import datetime, timedelta
from pymilvus import connections, Collection, utility
from milvus import MilvusClient
from dotenv import load_dotenv

load_dotenv(".env", override=True)


def init_milvus():
    client = MilvusClient(uri=os.getenv("MILVUS_CLIENT"))

    milvus_client = os.getenv("MILVUS_CLIENT")
    pattern = r'http://([\d.]+):(\d+)'
    match = re.search(pattern, milvus_client)
    if not match:
        raise ValueError("未匹配到有效的 Milvus URL 格式")
    host, port = match.group(1), match.group(2)

    db_name = os.getenv("MILVUS_DB_NAME")
    databases = client.list_databases()
    if db_name not in databases:
        client.create_database(db_name=db_name)
        print(f"创建数据库: {db_name}")

    connections.connect(
        alias="default",
        db_name=db_name,
        host=host,
        port=port
    )
    print(f"已连接到 Milvus 数据库: {db_name} (host: {host}, port: {port})")
    return client, db_name


def check_disk_usage(path="/ai"):
    disk_stats = shutil.disk_usage(path)
    used_percent = (disk_stats.used / disk_stats.total) * 100
    print(f"{path} 磁盘使用率: {used_percent:.2f}%")
    return used_percent


def delete_last_day_data():
    collections = utility.list_collections()
    if not collections:
        print("没有可用的集合，无需清理")
        return

    one_day_ago = datetime.now() - timedelta(days=1)
    time_field = "capture_time"

    for coll_name in collections:
        coll = Collection(coll_name)
        coll.load()

        schema = coll.schema
        fields = [f.name for f in schema.fields]
        if time_field not in fields:
            print(f"集合 {coll_name} 不包含时间字段 {time_field}，跳过清理")
            continue
        if "image_url" not in fields and "large_image_url" not in fields:
            print(f"集合 {coll_name} 不包含 image_url 或 large_image_url 字段，跳过清理")
            continue

        one_day_ago_ts = int(one_day_ago.timestamp() * 1000)
        expr = f"{time_field} >= {one_day_ago_ts}"

        result = coll.query(expr=expr, output_fields=["count(*)"])
        count = result[0]["count(*)"] if result else 0
        if count == 0:
            print(f"集合 {coll_name} 中最后一天无数据，无需删除")
            continue

        print(f"集合 {coll_name} 中最后一天有 {count} 条数据，开始删除...")
        delete_result = coll.delete(expr=expr)
        print(f"集合 {coll_name} 删除完成，删除条数: {delete_result.delete_count}")

        coll.release()


def main():
    try:
        client, db_name = init_milvus()
    except Exception as e:
        print(f"Milvus 初始化失败: {e}")
        return

    disk_usage = check_disk_usage("/ai")
    if disk_usage > 98:
        print(f"磁盘使用率超过 98%（当前 {disk_usage:.2f}%），开始执行清理...")
        try:
            delete_last_day_data()
            print("清理完成，再次检查磁盘使用率:")
            check_disk_usage("/ai")
        except Exception as e:
            print(f"清理过程出错: {e}")
    else:
        print(f"磁盘使用率未超过阈值（98%），无需清理")

    connections.disconnect("default")
    print("已断开 Milvus 连接")


if __name__ == "__main__":
    main()