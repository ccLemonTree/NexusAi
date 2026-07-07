import os
import requests
import json

# 配置（改这3行就行）
IMAGE_FOLDER = r"D:\脚本大本营\runs\detect\person"  # 图片文件夹路径
API_URL = "http://192.168.1.118:10012/ai/seefor/api/insert/testvec2milvus"  # 接口地址
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')  # 支持的图片格式

# 遍历文件夹+调用接口
for root, _, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if file.lower().endswith(SUPPORTED_EXT):
            pic_path = os.path.join(root, file)
            # 调用接口
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"pic_path": pic_path.replace("\\", "\\\\")})  # 转义反斜杠
            )
            # 简单打印结果
            print(f"{pic_path} -> {response.status_code} | {response.text[:50]}")  # 只显示前50个字符
