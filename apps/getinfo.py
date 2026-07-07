from api.register_model.register import Resgister_Template
from fastapi import APIRouter

router = APIRouter()

@router.get("/template")
async def getTemplate():
    message = "获取成功"
    templates = {}
    try:
        templates = Resgister_Template()()
    except Exception as e:
        message = e
    finally:
        return {
            "data": templates,
            "message": message
        }