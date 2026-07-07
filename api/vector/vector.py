import os
import io
import base64
import asyncio
from typing import Optional, List, Tuple, Literal

import numpy as np
import cv2
from PIL import Image
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.create_embedding_response import CreateEmbeddingResponse

from tools.logger_tools import app_logger

logger = app_logger.getChild("vector")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = "/models/qwen3-vl-4bit-gptq"
DEFAULT_INSTRUCTION = "Represent the user's input."

_async_client: Optional[AsyncOpenAI] = None


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
    return _async_client


def _encode_image_to_data_uri(image_data) -> str:
    if isinstance(image_data, np.ndarray):
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    elif isinstance(image_data, str):
        if not os.path.exists(image_data):
            raise FileNotFoundError(f"图片路径不存在: {image_data}")
        pil_img = Image.open(image_data).convert("RGB")
    elif isinstance(image_data, Image.Image):
        pil_img = image_data.convert("RGB")
    else:
        raise TypeError(f"不支持的图片数据类型: {type(image_data)}")

    if pil_img.width > 2048 or pil_img.height > 2048:
        pil_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def create_chat_embeddings(
        client: AsyncOpenAI,
        *,
        messages: list,
        model: str,
        encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
) -> CreateEmbeddingResponse:
    return client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={
            "messages": messages,
            "model": model,
            "encoding_format": encoding_format,
            "continue_final_message": continue_final_message,
            "add_special_tokens": add_special_tokens,
        },
    )


async def gme_vector(
        question: Optional[str] = None,
        prompt: Optional[str] = None,
        pic_path=None,
        search_type: bool = False,
        insert_type: str = ""
) -> Tuple[List[np.ndarray], str]:
    client = _get_async_client()
    if question is None and pic_path is None:
        raise ValueError("question 和 pic_path 不能同时为空")

    instruction = prompt or DEFAULT_INSTRUCTION
    emb_type = "text" if pic_path is None else "picture"

    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            }
        ]

        user_content = []

        if pic_path is not None:
            if isinstance(pic_path, str) and pic_path.startswith(('http://', 'https://', 'file://')):
                img_url = pic_path
            else:
                img_url = await asyncio.to_thread(_encode_image_to_data_uri, pic_path)
            user_content.append({"type": "image_url", "image_url": {"url": img_url}})

        if question is not None:
            user_content.append({"type": "text", "text": question})

        if pic_path is not None and question is None:
            user_content.append({"type": "text", "text": ""})

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})

        resp = await create_chat_embeddings(
            client,
            messages=messages,
            model=VLLM_MODEL,
            encoding_format="float",
            continue_final_message=True,
            add_special_tokens=True,
        )

        vector = resp.data[0].embedding
        logger.info("[向量] 成功 | 类型=%s | 维度=%d", insert_type or emb_type, len(vector))
        return vector, emb_type

    except Exception as e:
        logger.error("[向量] 失败 | 类型=%s | 错误: %s", insert_type or emb_type, e, exc_info=True)
        raise RuntimeError(f"向量请求失败: {e}") from e
