
from fastapi import APIRouter,Request
import numpy as np
import json, cv2, logging
from api.infer.Utils.coordTransform_py.Pnp import pnp
logger = logging.getLogger("Analyse")
router = APIRouter()

@router.post('/matrix')
async def pnpMatrix(request: Request):
    '''
    {
  "demarcateSaveList": [
    {
      "pans": 1916,
      "tilt": 3403,
      "zoom": 0,
      "wgs84Latitude": 29.49093,
      "wgs84Longitude": 121.88558

    },    {
      "pans": 2573,
      "tilt": 3476,
      "zoom": 0,
      "wgs84Latitude": 29.48779,
      "wgs84Longitude": 121.88621

    },    {
      "pans": 3035,
      "tilt": 3405,
      "zoom": 0,
      "wgs84Latitude": 29.48817,
      "wgs84Longitude": 121.88335

    },    {
      "pans": 1159,
      "tilt": 3514,
      "zoom": 0,
      "wgs84Latitude": 29.49465,
      "wgs84Longitude": 121.88318

    }
  ],
  "captureType": "CAPTURE_SERVICE_DAHUATECH_SDK"
}
    '''
    data = await request.body()
    matrixInfo = json.loads(data)
    ptzs = []
    jwds = []
    type = matrixInfo["captureType"]
    for data in matrixInfo['demarcateSaveList']:
        ptzs.append([data['pans'], data['tilt'], data['zoom']])
        jwds.append([data['wgs84Longitude'], data['wgs84Latitude']])
    flag = True
    msg = "Successful"
    vector_rotation, vector_translation ,images_points= [], [],[]
    try:
        jwds, ptzs = pnp.dataFormat(jwds, ptzs, type)
        # pt 转平面
        images_points = pnp.ptz_touv(ptzs)
        # 平面 转世界坐标系
        images_points = pnp.uv_toworld(pnp.camera_matrix, images_points)
        # 经纬度转世界坐标系
        world_points = pnp.jwd_towordld(jwds)
        vector_rotation, vector_translation = pnp.solvePNP(world_points, images_points, pnp.camera_matrix)
        vector_translation = vector_translation.tolist()
        vector_rotation = vector_rotation.tolist()
        images_points = images_points.tolist()
    except Exception as e:
        flag = False
        msg = f"{e}"
    return json.dumps({
        "checkStatus": flag,
        "msg": msg,
        "rotationVector": f"{vector_rotation}",
        "translationVector": f"{vector_translation}",
        "vectorWorld":f"{images_points}"
    })

@router.post('/ptzlocation')
async def pnpPtzlocation(request: Request):
    flag = True
    msg = ""
    wgs84Longitude, wgs84Latitude = 0,0
    data = await request.body()
    matrixInfo = json.loads(data)
    type = matrixInfo["captureType"]
    rotationVector = matrixInfo['rotationVector']
    translationVector = matrixInfo['translationVector']
    p = matrixInfo['pans']
    t = matrixInfo['tilt']
    z = matrixInfo['zoom']
    if pnp.dtype[type]:
        p = 3600-p
    rotationVector = np.array(json.loads(rotationVector))
    translationVector = np.array(json.loads(translationVector))
    vector_rotation_R, _ = cv2.Rodrigues(rotationVector)
    try:
        wgs84Longitude, wgs84Latitude = pnp.pt_tojwd((p, t, z), translationVector, vector_rotation_R)
    except Exception as e:
        logger.info(e)
        flag = False
        msg = f"{e}"
    return json.dumps({
        "checkStatus": flag,
        "msg": msg,
        "wgs84Longitude": wgs84Longitude,
        "wgs84Latitude": wgs84Latitude
    })


@router.post('/LLatlocation')
async def pnpLLatlocation(request: Request):
    flag = True
    msg = ""
    pans, tilt = 0, 0
    data = await request.body()
    matrixInfo = json.loads(data)
    rotationVector = matrixInfo['rotationVector']
    translationVector = matrixInfo['translationVector']
    images_points = matrixInfo['vectorWorld']
    type = matrixInfo["captureType"]
    wgs84Longitude = matrixInfo['wgs84Longitude']
    wgs84Latitude = matrixInfo['wgs84Latitude']
    pans = matrixInfo['pans']
    tilt = matrixInfo['tilt']
    zoom = matrixInfo['zoom']
    ptzs = [[pans,tilt]]
    rotationVector = np.array(json.loads(rotationVector))
    translationVector = np.array(json.loads(translationVector))
    images_points = np.array(json.loads(images_points))
    vector_rotation_R, _ = cv2.Rodrigues(rotationVector)
    try:
        pans ,tilt = pnp.jwd_topt((wgs84Longitude, wgs84Latitude),rotationVector, translationVector, vector_rotation_R, images_points,type,ptzs)
    except Exception as e:
        logger.info(e.__traceback__.tb_frame.f_globals["__file__"])
        logger.info(e.__traceback__.tb_lineno)
        logger.info(e)
        flag=False
        msg = f"{e}"
    return json.dumps({
        "checkStatus": flag,
        "msg": msg,
        "pans": int(pans),
        "tilt": int(tilt),
        "zoom": 0
    })
