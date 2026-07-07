from PIL import Image
import io
from fastapi import File,Body,Form,UploadFile,FastAPI,Depends, HTTPException
from pydantic import BaseModel, Field
from gme_inference import GmeQwen2VL
import numpy as np
import uvicorn
import time
import logging
from typing import List, Optional
import torch


#option={}
#option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="GmeQwen2VL图像向量服务", description="基于GmeQwen2VL模型的图像向量生成服务")

# 模型路径配置
MODEL_PATH = "gme-Qwen2-VL-2B-Instruct"

# 全局模型实例
model = None

from typing import Optional, List


# 定义 JSON 请求体模型
class EmbeddingRequest(BaseModel):
    question: Optional[str] = None
    prompt: str = "You are a helpful assistant."
# 请求模型
#class ImageEmbeddingRequest(BaseModel):
#    question: str = Field(None, description="文本描述")
#    prompt: str =  Field('You are a helpful assistant.', description="system 提示词")
    

# 启动时加载模型
@app.on_event("startup")
async def load_model():
    global model
    try:
        start_time = time.time()
        logger.info(f"开始加载模型: {MODEL_PATH}")
        model = GmeQwen2VL(MODEL_PATH)
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


# 图像向量生成接口 - JSON参数
@app.post("/v1/embeddings",response_model=List[List[float]])
async def get_image_embeddings(
        file: Optional[UploadFile] = File(None), #file: UploadFile = File(...),
        question: Optional[str] = Form(None),
        prompt: str = Form("You are a helpful assistant.")):
    """通过JSON参数获取图像向量"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        print(question)
        # 记录开始时间
        start_time = time.time()
        # 读取上传的文件内容
        pil_image= None
        if file is not None:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        vector = model.embed(images=[pil_image], texts=[question], instruction=prompt).cpu()
        # 计算耗时
        inference_time = time.time() - start_time

        # 记录结果
        vector_shape = np.array(vector, dtype=np.float32)

        logger.info(f"图像向量生成成功，形状: {vector_shape}, 耗时: {inference_time:.2f}秒")
        return [list(vector_shape[0])]
        '''return {
            "vector": list(vector_shape[0]),
            "inference_time": inference_time,
            "message": "成功生成图像向量"
        }'''


    except Exception as e:
        logger.error(f"图像向量生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 主函数
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8997)    
