# -*- coding: utf-8 -*-
"""
终极极简违停 —— 只靠底图文件时间判断
逻辑：
1. 看底图修改时间是否超过配置时间
2. 没超过 → 直接返回空
3. 超过了 → 底图跑AI + 当前帧AI做IOU=0.92匹配
4. 匹配上 = 车没动 = 违停
5. 违停才更新底图，不违停不更新
"""
import os
import cv2
from datetime import datetime
from tools.init import cfg
from api.infer.Utils.class_info import ModelClass
from api.infer.running import analyseRun
from api.infer.Utils.result_utils import iou  # 你项目里自带的IOU函数


class Model(ModelClass):
    def execute(self):
        # 设备信息
        deviceid = self.cameraInfo.deviceId
        presetid = self.cameraInfo.presetId
        logic_name = self.logicModelName

        # 配置时间
        check_time = int(self.labelRules.get(logic_name, {}).get("time", 300))

        # 底图路径
        save_dir = os.path.join(cfg.root_path, f"Picture_pkl/{deviceid}/{presetid}/{logic_name}")
        save_dir = f"E:/Picture_pkl/{deviceid}/{presetid}/{logic_name}"

        os.makedirs(save_dir, exist_ok=True)
        bg_path = os.path.join(save_dir, "bg.jpg")

        result_list = []
        img_list = []
        now = datetime.now()

        # ===================== 时间没到，直接返回空 =====================
        if os.path.exists(bg_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(bg_path))
            if (now - file_time).seconds < check_time:
                return [], []

        # ===================== 时间到了，开始判断 =====================
        # 没有底图（第一次）→ 保存底图，返回空
        if not os.path.exists(bg_path):
            cv2.imwrite(bg_path, self.picture)
            return [], []

        # 1. 读取底图并跑AI分析
        labels = cfg.logicModelDict[self.logicModelName][0]['label']
        cut_img = cv2.imread(bg_path)
        if cut_img is None:
            cv2.imwrite(bg_path, self.picture)
            return [], []

        # 底图跑模型
        bg_result = analyseRun(labels, [cut_img, cut_img], self.cameraInfo)

        # 2. 【核心】当前帧目标 和 底图目标 做 IOU 0.92 匹配
        result_true = []
        for current_obj in self.logicResult:
            cur_box = [current_obj.x1, current_obj.x2, current_obj.y1, current_obj.y2]
            # 和底图所有目标匹配
            for bg_obj in bg_result:
                bg_box = [bg_obj.x1, bg_obj.x2, bg_obj.y1, bg_obj.y2]
                # IOU 超过 0.92 = 完全没动
                if iou(cur_box, bg_box) >= 0.92:
                    result_true.append(current_obj)
                    break

        # 3. 匹配上的目标 = 违停
        for obj in result_true:
            obj.classname = logic_name
            result_list.append(obj)

        # 4. 有违停 → 返回图片 + 更新底图
        if result_list:
            img_list = [self.pic2base64(self.picture)]
            cv2.imwrite(bg_path, self.picture)

        return result_list, img_list