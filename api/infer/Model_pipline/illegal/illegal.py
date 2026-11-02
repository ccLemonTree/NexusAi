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
import numpy as np
from datetime import datetime
from tools.init import cfg
from api.infer.Utils.class_info import ModelClass
from api.infer.running import analyseRun
from api.infer.Utils.result_utils import iou  # 你项目里自带的IOU函数
from tools.logic_state import atomic_write, locked_state


class Model(ModelClass):
    def execute(self):
        logic_name = self.logicModelName

        # 配置时间
        check_time = int(self.labelRules.get(logic_name, {}).get("time", 300))

        with locked_state(self.cameraInfo, logic_name) as save_dir:
            bg_path = save_dir / "bg.jpg"
            result_list = []
            img_list = []
            cut_img = None

            try:
                # Open before reading metadata so NFS close-to-open consistency applies.
                with bg_path.open("rb") as source:
                    file_time = datetime.fromtimestamp(os.fstat(source.fileno()).st_mtime)
                    if (datetime.now() - file_time).total_seconds() < check_time:
                        return [], []
                    contents = source.read()
                    if contents:
                        cut_img = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
            except FileNotFoundError:
                pass

            if cut_img is not None:
                labels = cfg.logicModelDict[self.logicModelName][0]['label']
                bg_result = analyseRun(labels, [cut_img, cut_img], self.cameraInfo)
                for current_obj in self.logicResult:
                    cur_box = [current_obj.x1, current_obj.x2, current_obj.y1, current_obj.y2]
                    for bg_obj in bg_result:
                        bg_box = [bg_obj.x1, bg_obj.x2, bg_obj.y1, bg_obj.y2]
                        if iou(cur_box, bg_box) >= 0.92:
                            current_obj.classname = logic_name
                            result_list.append(current_obj)
                            break

            if cut_img is None or result_list:
                success, encoded = cv2.imencode(".jpg", self.picture)
                if not success:
                    raise ValueError("Failed to encode illegal parking baseline")
                atomic_write(bg_path, encoded.tobytes())
            if result_list:
                img_list = [self.pic2base64(self.picture)]

            return result_list, img_list
