from fastapi import APIRouter
from api.vector.vector import gme_vector
from api.search.search import research
from pydantic import BaseModel
from tools.init import client
from tools.logger_tools import CangQiong_Smart_Search_Logger as logger

router = APIRouter()

class SearchInputRequestData(BaseModel):
    text: str = ""
    pic_path: str = None
    device_ids: list = None
    start_time: str = ""
    end_time: str = ""
    limit: int = 15


@router.post("/vlm2vector")
async def vlm2vector(request: SearchInputRequestData):
    results = []
    message = "查询成功"
    if request.pic_path is None and request.text is None:
        return {"message": "参数不能为空"}
    try:
        vector, emb_type = await gme_vector(
            question=request.text,
            pic_path=request.pic_path,
            search_type=True
        )
        logger.info(f"{request.dict()}")
        results = research(
            client,
            request.text,
            vector,
            filter_divice=request.device_ids,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit,
            emb_type=emb_type
        )
    except Exception as e:
        message = (
            f"查询失败：{e} \n"
            f"保存内容：{e.__traceback__.tb_frame.f_globals['__file__']}\n"
            f"报错行数：{e.__traceback__.tb_lineno}"
        )
        logger.error(message)
    return {'data': results, 'message': message}
