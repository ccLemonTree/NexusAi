import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# ===================== 配置 =====================
IMAGE_DIR = r"D:\DATA\dwcar\datas\b\images"
API_URL = "http://192.168.1.118:10012/ai/seefor/api/insert/vec2milvus"
HEADERS = {"Content-Type": "application/json"}

# 并发线程数（根据你的接口承受能力调，10～50都可以）
MAX_WORKERS = 30

# 固定参数
BASE_DATA = {
    "capture_time": 1437537,
    "device_name": "str",
    "device_id": "str",
    "channel_id": 456678,
    "channel_name": "str",
    "channel_number": "str"
}

# 支持的图片格式
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG')
# =================================================


def insert_one(pic_path, index, total):
    try:
        data = BASE_DATA.copy()
        data["pic_path"] = pic_path

        resp = requests.post(API_URL, json=data, headers=HEADERS, timeout=15)
        print(f"✅ 成功 [{index}/{total}] | {os.path.basename(pic_path)} | 状态码：{resp.status_code}")
    except Exception as e:
        print(f"❌ 失败 [{index}/{total}] | {os.path.basename(pic_path)} | 错误：{str(e)}")


def batch_insert_concurrent():
    # 读取所有图片
    img_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(IMG_EXTS)]
    total = len(img_paths)
    print(f"=" * 60)
    print(f"🚀 开始并发插入，总图片数：{total}")
    print(f"🚀 并发数：{MAX_WORKERS}")
    print(f"=" * 60)

    start = time.time()

    # 多线程并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, path in enumerate(img_paths, 1):
            executor.submit(insert_one, path, i, total)

    end = time.time()
    print(f"\n🏁 全部插入完成！耗时：{end - start:.2f} 秒")


if __name__ == '__main__':
    batch_insert_concurrent()