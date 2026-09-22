import asyncio
import os
import time
from typing import Optional

import cv2
import numpy as np

from tools.logger_tools import CangQiong_Smart_Vllm_logger as logger_vllm


_eos_session = None
_eos_client = None
_eos_client_lock = asyncio.Lock()


def _validate_eos_config():
    endpoint = os.getenv("EOS_ENDPOINT", "")
    ak = os.getenv("EOS_ACCESS_KEY", "")
    sk = os.getenv("EOS_SECRET_KEY", "")
    bucket = os.getenv("EOS_BUCKET", "")
    missing = [name for name, value in {
        "EOS_ENDPOINT": endpoint,
        "EOS_ACCESS_KEY": ak,
        "EOS_SECRET_KEY": sk,
        "EOS_BUCKET": bucket,
    }.items() if not value]
    if missing:
        raise EnvironmentError(f"EOS 配置缺失: {', '.join(missing)}。请在 .env 或环境变量中设置。")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint, ak, sk, bucket


async def _get_eos_client(max_retries: int = 2):
    global _eos_session, _eos_client
    if _eos_client is not None:
        return _eos_client
    try:
        import aioboto3
        from botocore.config import Config as BotocoreConfig
    except ImportError as e:
        raise RuntimeError("EOS 依赖未安装，请重建镜像以安装 aioboto3") from e

    async with _eos_client_lock:
        if _eos_client is not None:
            return _eos_client
        endpoint, ak, sk, bucket = _validate_eos_config()
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                _eos_session = aioboto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk)
                _eos_client = await _eos_session.client(
                    "s3",
                    endpoint_url=endpoint,
                    config=BotocoreConfig(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}, max_pool_connections=50),
                ).__aenter__()
                await _eos_client.head_bucket(Bucket=bucket)
                return _eos_client
            except Exception as e:
                last_error = e
                _eos_client = None
                _eos_session = None
                logger_vllm.warning(f"EOS 客户端初始化失败 (尝试 {attempt}/{max_retries}): [{type(e).__name__}] {e}")
        raise ConnectionError(f"EOS 客户端初始化失败，已重试 {max_retries} 次: {last_error}")


async def load_eos_image(key: str) -> Optional[np.ndarray]:
    timeout = int(os.getenv("EOS_TIMEOUT", "15"))
    bucket = os.getenv("EOS_BUCKET", "")
    eos_url = f"{os.getenv('EOS_ENDPOINT', '').rstrip('/')}/{bucket}/{key.lstrip('/')}"
    started = time.monotonic()
    try:
        s3 = await _get_eos_client()
        response = await asyncio.wait_for(
            s3.get_object(Bucket=bucket, Key=key),
            timeout=timeout,
        )
        data = await asyncio.wait_for(response["Body"].read(), timeout=timeout)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode 返回 None，可能不是合法图片")
        logger_vllm.info(f"EOS 图片下载成功 耗时={time.monotonic() - started:.3f}s size={len(data)}B key={key[:120]}")
        return image
    except asyncio.TimeoutError:
        logger_vllm.error(f"EOS 图片下载超时(>{timeout}s) eos_url={eos_url}")
    except Exception as e:
        logger_vllm.error(f"EOS 图片下载失败 耗时={time.monotonic() - started:.3f}s eos_url={eos_url}: [{type(e).__name__}] {e}")
    return None
