from dotenv import load_dotenv
load_dotenv(".env", override=True)
import os
import urllib
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from apps import chat
from apps import getinfo
from apps import register
from apps.Face_Search import face_search
from apps.Cangqiong_Smart_Search import insert, search
from apps.Cangqiong_Smart_Search import config as search_config
from apps.Cangqiong_Smart_Analyse import analyse
from apps.Cangqiong_Smart_Analyse import config as analyse_config
from apps.Alarm_Location import location
from tools.logger_tools import app_logger, request_logger, error_logger, configure_logging

# 以节点形式的路由
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/config/reload")
def start_init():
    load_dotenv(".env", override=True)
    return {"message": "config reloaded"}


_IMAGE_ROOT = os.path.normpath(os.getenv("IMAGE_ROOT_DIR", "/ai"))

@app.get("/image/{file_path:path}")
async def get_image(file_path: str):
    decoded_path = urllib.parse.unquote(file_path)
    normalized_path = os.path.normpath(decoded_path)
    if not normalized_path.startswith(_IMAGE_ROOT):
        raise HTTPException(status_code=403, detail="访问路径不在允许范围内")
    if not os.path.isfile(normalized_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(normalized_path)

app.include_router(chat.router, prefix="/ai/seefor/api/chat", tags=["Chat"])
app.include_router(getinfo.router, prefix="/ai/seefor/api/getinfo", tags=["Getinfo"])
app.include_router(register.router, prefix="/ai/seefor/api/register", tags=["Register"])

#人脸检索
app.include_router(face_search.router, prefix="/ai/seefor/api/information", tags=["Face Search"])

# 苍穹智搜
app.include_router(insert.router,prefix="/ai/seefor/api/insert",tags=["insert"])
app.include_router(search.router,prefix="/ai/seefor/api/search",tags=["search"])
app.include_router(search_config.router,prefix="/ai/seefor/api/search/config",tags=["config"])

# 苍穹智眸
app.include_router(analyse.router,prefix="/ai/seefor/api/analyse",tags=["analyse"])
app.include_router(analyse_config.router,prefix="/ai/seefor/api/config",tags=["get_modelconfig"])


# 苍穹智脑

# 预警定位
app.include_router(location.router,prefix="/ai/seefor/api/calibration",tags=["location"])

configure_logging(app)





if __name__ == '__main__':
    import uvicorn

    print("应用启动成功，可用路由：")
    for route in app.routes:
        if hasattr(route, "methods"):  # 过滤非HTTP路由（如WebSocket）
            print(f"  - {route.path} [{', '.join(sorted(route.methods))}]")
    uvicorn.run(app, host="0.0.0.0", port=10012)
