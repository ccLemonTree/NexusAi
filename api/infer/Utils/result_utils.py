# -*- coding: utf-8 -*-
import importlib
import time, os
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.Utils.analyse_utils import *
from concurrent.futures import as_completed
from api.infer.Utils.boundingbox import BoundingBox
import asyncio

@analyseTime
def Unlogic_run(img, model_dicts, triton, executor,label_rules=None, cameraInfo=None, box_info=BoundingBox):
    """
    :param img:         图片名称
    :param model:       模型字典  {'ele_nohalmet': [200, 202]}}
    :param cfg:         文件配置类
    :param roi_dict:    roi 以及 nroi 框
    :param triton:      模型服务类
    :return:
    """
    # 线程列表
    futureList = []
    for servername, analyselabel in model_dicts.items():  # yolov5_person  , [dog,person]
        analyselabel_rules = {}  # 每个模型独立初始化，避免前一模型的标签规则泄漏到后续模型
        if isinstance(label_rules,dict):
            for key,value in label_rules.items():
                if key in analyselabel:
                    analyselabel_rules[key] = value
        elif isinstance(label_rules,list):
            analyselabel_rules = label_rules
        else:
            analyselabel_rules = analyselabel
        futureList.append(executor.submit(triton.run, servername, img, cameraInfo, label_to_detect=analyselabel_rules,
                                          box_info=box_info))  # 提交任务
    # 结果无续返回
    boundingList = []
    for future in as_completed(futureList):
        futureResult = future.result()
        boundingList += futureResult
    return boundingList


# @analyseTime
# def Unlogic_run(img, model_dicts, triton, executor,label_rules=None, cameraInfo=None, box_info=BoundingBox):
#     """
#     :param img:         图片名称
#     :param model:       模型字典  {'ele_nohalmet': [200, 202]}}
#     :param cfg:         文件配置类
#     :param roi_dict:    roi 以及 nroi 框
#     :param triton:      模型服务类
#     :return:
#     """
#     # 线程列表
#     futureList = []
#     analyselabel_rules = {}
#     for servername, analyselabel in model_dicts.items():  # yolov5_person  , [dog,person]
#         if isinstance(label_rules,dict):
#             for key,value in label_rules.items():
#                 if key in analyselabel:
#                     analyselabel_rules[key] = value
#         elif isinstance(label_rules,list):
#             analyselabel_rules = label_rules
#         else:
#             analyselabel_rules = analyselabel
#         futureList.append(executor.submit(triton.run, servername, img, cameraInfo, label_to_detect=analyselabel_rules,
#                                           box_info=box_info))  # 提交任务
#     # 结果无续返回
#     boundingList = []
#     for future in as_completed(futureList):
#         futureResult = future.result()
#         boundingList += futureResult
#     return boundingList



# @analyseTime
def logic_run(imgs, logicResults, camerInfo, logicModelName, tritonServer,label_rules):
    """
        picture, camera_info, model, cfg, boundingboxs,
    :param picture: 图片名称
    :param camera_info: 设备信息 deviceid and preset  type={}
    :param model: 模型名称
    :param eve: 事件类型
    :param cfg: 配置文件
    :param roi_dict: roi 框
    :param boundingboxs: 目标框
    :param triton: triton 推理 类
    :param objclasses: 自定义推理类  # 去除
    :return:
    """
    Model_type = logicModelName.split('-')[0]
    images_list = []
    model_pipline = importlib.import_module("api.infer.Model_pipline.{}.{}".format(Model_type, Model_type))
    try:
        c = model_pipline.Model(imgs, logicResults, camerInfo, logicModelName, tritonServer,label_rules)
        result, images_list = c.execute()
        # result, images_list = asyncio.run(c.execute())
    except Exception as e:
        result = {}
        logger.error(f"{logicModelName} {e}")
        logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
        logger.error(e.__traceback__.tb_lineno)

    return {
        logicModelName: {
            "bbox": result, #[i.dict() for i in result],
            "imglist": images_list,
        }
    }


def modelpipline_cfg(step, model_cfg, **kwargs):
    """
    根据传入的 step number  进行 输入和输出
    :param step:
    :return:
    """
    input = []
    output = []
    inputkey = {}
    outputkey = {}
    step_info = model_cfg['ensemble_scheduling']['dog&line']['step1']
    models = step_info.get('model')
    step_type = step_info.get('type')
    loadinput = step_info.get('input')
    loadoutput = step_info.get('output')
    for i in loadinput:
        for j in i['input_map']:
            inputkey[j['key']] = j['value']
    for i in loadoutput:
        for j in i['output_map']:
            outputkey[j['key']] = j['value']


def iou(box1, box2):
    '''
    将符号坐标和模板坐标放入IOU进行匹配
    :param box1: 符号坐标列表
    :param box2: 模板坐标列表
    :return: 交并比结果 题目和符号重合的地方占整个符号的多少
    '''
    h = max(0, min(box1[1], box2[1]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[2], box2[2]))
    area_box1 = ((box1[1] - box1[0]) * (box1[3] - box1[2]))
    inter = w * h
    iou = inter / area_box1
    return iou
