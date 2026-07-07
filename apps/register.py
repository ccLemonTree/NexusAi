from typing import Dict, Any
from api.register_model.register import Resgister_Model
from fastapi import APIRouter

router = APIRouter()


@router.post("/config/model/add")
async def addModel(data: Dict[str, Any]):
    message = "请核对参数"
    try:
        message = Resgister_Model(data)()
    except Exception as e:
        message = e
    finally:
        return {"message": message}


@router.post("/config/model/check")
async def checkModel(data: Dict[str, Any]):
    message = "请核对参数"
    try:
        message = Resgister_Model(data)()
    except Exception as e:
        message = e
    finally:
        return {"message": message}
