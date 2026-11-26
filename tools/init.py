import os
from functools import lru_cache

from pymilvus import MilvusClient,Function, FunctionType
from api.infer.Triton_model.triton_client import triton_inference
from api.infer.openai_infer import openai_infer
from api.infer.fire_infer import fire_infer
from api.infer.mindie_infer import mindie_infer
from api.infer.Utils.config import Cfg

infer_backend = {
    "mindie":mindie_infer,
    "openai":openai_infer,
    "fire":fire_infer
}


vector_search_enabled = os.getenv("VECTOR_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
client = MilvusClient(uri=os.getenv("MILVUS_CLIENT"), db_name=os.getenv("MILVUS_DB_NAME")) if vector_search_enabled else None
chat_infer = infer_backend[os.getenv("INFER_BACKEND")](os.getenv("STRUCT_EMBEDDING_MODEL"), os.getenv("STRUCT_EMBEDDING_MODEL_NAME"))
tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[os.getenv("TRITON_SERVER")])
cfg = Cfg()


@lru_cache(maxsize=1)
def get_redis_client():
    """Return the Redis client shared by algorithms in this process."""
    host = os.getenv("REDIS_HOST")
    if not host:
        raise ValueError("REDIS_HOST is required to use Redis")

    from redis import Redis

    return Redis(
        host=host,
        port=int(os.getenv("REDIS_PORT") or 6379),
        db=int(os.getenv("REDIS_DB") or 0),
        password=os.getenv("REDIS_PASSWORD") or None,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
