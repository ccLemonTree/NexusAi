# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2022年11月29日 14:33:13
@packageName Triton
@className illegal_bicycle
@version 1.0.0
@describe TODO
"""
import os.path
import uuid
from datetime import datetime
import pickle
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.Utils.result_utils import iou
from api.infer.Utils.utils import *
from tools.init import cfg
from api.infer.Utils.class_info import ModelClass


# from Utils.boundingbox import BoundingBox
class Model(ModelClass):
    def execute(self) -> [list, dict]:
        pictures = []
        cut_img = None
        # 存在违停的 data
        flag = False
        have_list = ["save.jpeg","saveold.jpeg"]
        result_list = []
        deviceid = self.cameraInfo.deviceId
        presetid = self.cameraInfo.presetId
        filename = os.path.join(cfg.root_path,f"Picture_pkl/{deviceid}/{presetid}/{self.logicModelName}")
        if os.path.isdir(filename) == False:
            os.makedirs(filename)
        dat_dirlist = os.listdir(filename)
        for i in have_list:
            if i in dat_dirlist:
                dat_dirlist.remove(i)
        # 没有 违停存在
        labelinfo = cfg.logicModelDict[self.logicModelName][0]['label']
        if len(dat_dirlist) == 0 and len(self.logicResult) > 0:
            logger.info(
                f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car  初次违停')
            # 循环所有 car 打包返回违停
            saveflag = False
            for info in self.logicResult:
                if info.classname in labelinfo:
                    uuids = uuid.uuid4()
                    saveflag = True
                    try:
                        cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                        cut_img = cv2.resize(cut_img, [235, 235])
                    except:
                        continue
                    des_bak = phash(cut_img)
                    with open(os.path.join(filename, str(uuids)), "wb") as f:
                        pickle.dump({"boxs": [info.x1, info.x2, info.y1, info.y2], "vector": des_bak,
                                     "updatetime": datetime.now(), "filename": str(uuids), "picture": cut_img}, f)
            if saveflag:
                if os.path.isfile(os.path.join(filename, "saveold.jpeg")):
                    os.remove(os.path.join(filename, "saveold.jpeg"))
                    os.rename(os.path.join(filename, "save.jpeg"), os.path.join(filename, "saveold.jpeg"))
                cv2.imwrite(os.path.join(filename, "save.jpeg"), self.picture)
            return [],pictures
        # 读取 所有dat
        dat_list = []
        for file in dat_dirlist:
            if os.path.isfile(os.path.join(filename, file)):
                with open(os.path.join(filename, file), "rb") as f:
                    dat_list.append(pickle.load(f))
        # 正式开始 违停逻辑匹配
        try:
            logger.info(
                f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car  遍历上次违停数据进行逻辑匹配{filename}/[{dat_dirlist}]')
            # 匹配到的 box
            matchIndex = [ ]
            for ids, info in enumerate(self.logicResult):
                if info.classname in labelinfo:
                    try:
                        cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                        cut_img = cv2.resize(cut_img, [235, 235])
                    except:
                        break
                    des_bak = phash(cut_img)
                    for jsobject, dat in enumerate(dat_list):
                        # 对iou大于0.8 的进行相似度匹配
                        if iou([info.x1, info.x2, info.y1, info.y2], dat['boxs']) > 0.85:
                            distance = bin(des_bak ^ dat['vector']).count('1')
                            similary = 1 - distance / max(len(bin(des_bak)), len(bin(des_bak)))
                            if similary > 0.6:
                                #logger.info(
                                #    f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car  违停相似度比对通过]{similary} > 0.6')
                                with open(os.path.join(filename, dat['filename']), "wb") as f:
                                    pickle.dump({"boxs": [info.x1, info.x2, info.y1, info.y2], "vector": des_bak,
                                                 "filename": dat['filename'], "picture": cut_img,
                                                 "updatetime": dat['updatetime']}, f)
                                    have_list.append(dat['filename'])
                                    csecond = (datetime.now() - dat['updatetime']).seconds
                                    btime = int(self.labelRules[self.logicModelName].get('time', 300))
                                    # if (datetime.now()-dat['updatetime']).seconds > int(cfg.logicModelDict[self.modelName][-1].get('time',300)):
                                    if (csecond > btime) and ((datetime.now() - datetime.fromtimestamp(
                                            os.path.getmtime(os.path.join(filename, "save.jpeg")))).seconds > btime):
                                        #logger.info(
                                        #    f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car  违停时间大于{btime}')
                                        info.classname = self.logicModelName
                                        result_list.append(info)
                                        flag = True
                                matchIndex.append(ids)
                                break


            for index,info in enumerate(self.logicResult):
                if index not in matchIndex:
                    try:
                        cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                        cut_img = cv2.resize(cut_img, [235, 235])
                    except:
                        continue
                    uuids = uuid.uuid4()
                    des_bak = phash(cut_img)
                    with open(os.path.join(filename, str(uuids)), "wb") as f:
                        pickle.dump({"boxs": [info.x1, info.x2, info.y1, info.y2], "vector": des_bak,
                                     "updatetime": datetime.now(), "picture": cut_img, "filename": str(uuids)},
                                    f)
                        have_list.append(str(uuids))
            # 违停重新计算
            images_list = []
            if flag:
                savejpeg = cv2.imread(os.path.join(filename, "save.jpeg"))
                images_list = [self.pic2base64(savejpeg)] # 违停前
                cv2.imwrite(os.path.join(filename, "saveold.jpeg"), savejpeg)
                cv2.imwrite(os.path.join(filename, "save.jpeg"), self.picture)
            for datname in os.listdir(filename):
                if datname not in have_list:
                    os.remove(os.path.join(filename, datname))
            return result_list,images_list
        except Exception as e:
            logger.error(
                f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car {e}')
            logger.error(
                f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car {e.__traceback__.tb_frame.f_globals["__file__"]}')
            logger.error(
                f'DeviceID [{self.cameraInfo.deviceId}] | Preset [{self.cameraInfo.presetId}] |  {self.cameraInfo.baseName} -> illegal-car {e.__traceback__.tb_lineno}')
            return [],[]
        finally:
            del cut_img
