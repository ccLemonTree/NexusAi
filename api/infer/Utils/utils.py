# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2022年11月30日 11:43:25
@packageName Triton
@className utils
@version 1.0.0
@describe TODO
"""
# -*- encoding=utf-8 -*-
import time,json
from functools import reduce
import os,re
import cv2
import pandas as pd
from PIL import Image
import psutil
import numpy as np
from api.infer.Utils.class_info import CameraInfo
# 这种算法的优点是简单快速，不受图片大小缩放的影响，
# 缺点是图片的内容不能变更。如果在图片上加几个文字，它就认不出来了。
# 所以，它的最佳用途是根据缩略图，找出原图。
from tools.init import cfg

# 格式化 json
def Json_load(analyse_data) -> CameraInfo:
    CameraInfo1 = CameraInfo()
    # 获取分析信息
    deviceId = analyse_data.get("deviceId", "")  # 设备信息
    presetId = analyse_data.get("presetId", "")  # 预置位信息
    picType = analyse_data.get("picType", "")    # 判断是2.5 还是3.0
    detectAreaId = analyse_data.get("detectId", "0")  # roi ID
    if picType == "":
        deviceId = analyse_data.get("deviceId", "")
        presetId = analyse_data.get("presetId", "")
        alarmTypeId = analyse_data.get("analyseT", "")
        picUrl = analyse_data.get("picUrl", "")
        picUrl2 = analyse_data.get("picUrl2", "")
        imgs_list = [picUrl,picUrl2]
        labels = analyse_data.get("labels", [])  # 事件ID
        picType = 0
        analyseROIs = eval(analyse_data.get("analyseROI", "").replace("true", "True").replace("null", "None").replace("false","False"))
        analyseROI = []
        for roi in analyseROIs:
            analyseROI.append({
                "detectAreaId": roi.get("detectId",0),
                "alarmTypeId": roi["analyseT"],
                "check": roi["check"],
                "points": roi["points"],
            })
    else:
        labels = analyse_data.get("labels", [])  # 事件ID
        alarmTypeId = analyse_data.get("alarmTypeId", 0)
        picture = analyse_data.get("picture", "")
        analyseROI = analyse_data.get("roi", [])
        imgs_list = picture
        if len(imgs_list)==1:
            picUrl = imgs_list[0]
            picUrl2 = picUrl
        elif len(imgs_list)>1:
            picUrl = imgs_list[0]
            picUrl2 = imgs_list[1]
        imgs_list = [picUrl,picUrl2]
    p = analyse_data.get("p", 0)
    t = analyse_data.get("t", 0)
    z = analyse_data.get("z", 0)
    # ------------- 结构化输出
    CameraInfo1.deviceId = deviceId
    CameraInfo1.presetId = presetId
    CameraInfo1.fileName1 = picUrl
    CameraInfo1.fileName2 = picUrl2
    CameraInfo1.analyseROI = analyseROI
    CameraInfo1.detectAreaId = detectAreaId
    CameraInfo1.alarmTypeId = alarmTypeId
    CameraInfo1.picType = picType
    match = re.search(r'snap(\d+)', os.path.basename(picUrl))
    CameraInfo1.snap = match.group(1) if match else 0
    CameraInfo1.imgsList = imgs_list
    CameraInfo1.ptz = {'ptz':{
        'p':p,
        't':t,
        'z':z
    }}
    CameraInfo1.defaultLabels = labels
    CameraInfo1.baseName = os.path.basename(picUrl)
    CameraInfo1.reserveParam = os.path.basename(picUrl)
    CameraInfo1.baseNameSplit = os.path.basename(os.path.splitext(CameraInfo1.baseName)[0])
    match2 = re.search(r'/dir(\d+)/', picUrl)
    CameraInfo1.dirName = match2.group(1) if match else 0

    return CameraInfo1



# 计算图片的局部哈希值--pHash
def phash(img):
    """
    :param img: 图片
    :return: 返回图片的局部hash值
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    img = img.resize((32, 32), Image.LANCZOS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / (1024)
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)

    return hash_value



def compare_images(image1, image2:list,filename,bigtime,conf):
    # 读取图片并转换为灰度图像
    try:
        des_bak = phash(image1)
        similarys = []
        j = 0
        for image in image2:
            des_bak2 = phash(image)
            distance = bin(des_bak ^ des_bak2).count('1')
            similary = 1 - distance / max(len(bin(des_bak)), len(bin(des_bak)))
            similarys.append(similary)
            j += 0
        index = np.argmax(similarys)
        cv2.imwrite(f"save/{filename}_first{bigtime}.jpeg", image1)
        for i, image in enumerate(image2):
            if i == index:
                cv2.imwrite(f"save/{filename}_first{bigtime}_sim{int(similarys[i] * 100)}_index.jpeg", image)
            else:
                cv2.imwrite(f"save/{filename}_first{bigtime}_sim{int(similarys[i] * 100)}.jpeg", image)
        if similarys[index] >= conf:
            return [similarys, index]
    except Exception as e:
        print(f"comp {e}")
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)
    return None,None

# 自定义计算两个图片相似度函数局部敏感哈希算法
def phash_img_similarity(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """

    # 读取图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # 计算两个图片的局部哈希值

    # 计算局部敏感哈希值
    img1_phash = phash(img1)
    img2_phash = phash(img2)

    # 打印局部敏感哈希值
    # 计算汉明距离
    distance = bin(img1_phash ^ img2_phash).count('1')
    similary = 1 - distance / max(len(bin(phash(img1))), len(bin(phash(img1))))
    return similary

def remove_file(file):
    if os.path.isfile(file):
        os.remove(file)



# 读取磁盘信息
def Read_disc(path):
    G = 1024 * 1024 * 1024
    disk = {}
    diskinfo = psutil.disk_usage(path)
    # 将字节转换成G
    disk['path'] = path
    disk['disk'] = diskinfo.total // G
    disk['diskUse'] = diskinfo.used // G
    disk['diskFree'] = diskinfo.free // G
    disk['diskRate'] = diskinfo.percent
    return disk


# 展开logic 的label：
def label_ravel(dicts):
    ravel = set()
    for i,value in dicts.items():
        if type(value) is dict:
            labels = value.get("label",None)
            if labels is None:
                break
            for j in labels:
                ravel.add(j)
    return list(ravel)

# 标签到 excel 输出
def create_labeltoexcel(all_lab):
    excel_path = os.path.join(cfg.root_path,"Config/remark.xlsx")
    res = pd.DataFrame(columns=["name", "remark","modeltype","modelname"],index=[data["label"] for data in all_lab])
    # 添加两行数据
    ressets = set(res.index)
    for j,datas in enumerate(all_lab,start=1):
        if datas["label"] in ressets:
            res.loc[datas["label"]]= [datas["name"],datas["remark"],datas["type"],datas["model"]]
            ressets.remove(datas["label"])
    # 如果文件本身存在 则继续添加
    if os.path.isfile(excel_path):
        res = pd.read_excel(excel_path,index_col=0)
        for j, datas in enumerate(all_lab):
            if datas["label"] in list(res.index):
                res.loc[datas["label"]] = [datas["name"],datas["remark"],datas["type"],datas["model"]]
    res.to_excel(os.path.join(cfg.root_path, "Config/remark.xlsx"))


def imgRead(cameraInfo):
    picType = cameraInfo.picType
    img = None
    images_list = []
    for fileName in cameraInfo.imgsList:
        if picType=="url":
            import cv2 as cv
            import urllib.request as request
            import numpy as np
            response = request.urlopen(fileName)
            img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            img = cv.imdecode(img_array, -1)
        elif picType=="base64":
            import cv2
            import base64
            import numpy as np
            imgData = base64.b64decode(fileName)
            nparr = np.frombuffer(imgData, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elif picType=='file':
            import cv2
            img = cv2.imread(fileName)
        images_list.append(img)
    return images_list

def Cos_Similarity(features1, features2):
    a_norm = np.linalg.norm(np.array(features1))
    b_norm = np.linalg.norm(np.array(features2))
    cos = np.dot(features1, features2) / (a_norm * b_norm)
    return cos

def getLogicLabelson(model):
    labels = []
    for i in cfg.logicModelDict[model]:
        result = cfg.logicModelDict[model][i].get("label",None)
        if result is None:
            continue
        else:
            labels.append(result)
    labels = sum(labels,[])
    return labels

def UpdateDeviceStatus(deviceid,status='0'):
    data = None
    with open(os.path.join("Device_status", deviceid, "deviceInformation.json"), "r") as f:
        data = json.load(f)
        data['status'] = status
    with open(os.path.join("Device_status", deviceid, "deviceInformation.json"), "w") as f:
        f.write(json.dumps(data))

