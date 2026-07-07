import os

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


client = MilvusClient(uri=os.getenv("MILVUS_CLIENT"), db_name=os.getenv("MILVUS_DB_NAME"))
chat_infer = infer_backend[os.getenv("INFER_BACKEND")](os.getenv("STRUCT_EMBEDDING_MODEL"), os.getenv("STRUCT_EMBEDDING_MODEL_NAME"))
tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[os.getenv("TRITON_SERVER")])
cfg = Cfg()
