from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/completion")
async def chat_Completion(request: Request):
    pass

