# -*- coding: utf-8 -*-
# @Time    : 2024/8/14 13:47:19
# @Author  : 陈澔麟
# @File    : process.py
import numpy as np


class VehicleAttribute(object):
    def __init__(self, color_threshold=0.5, work_threshold=0.5, peng_threshold=0.4, fx_threshold=0.5):
        self.color_threshold = color_threshold
        self.work_threshold = work_threshold
        self.peng_threshold = peng_threshold
        self.fx_threshold = fx_threshold
        self.color_list = [
            # "hong", "cheng", "huang", "lan", "lv", "bai"
            "红","橙","黄","蓝","绿","白"
        ]
        self.work_list = [
            # "gongzuozhong", "meigongzuo"
            "工作中", "没工作"
        ]
        self.peng_list = [
            # "gailepengbu","wanquangaizhu","bangai","weigai"
            "盖了篷布","盖了盖板","半盖","未盖"
                          ]
        self.fx_list = [
            # "shang","xia","zuo","you"
            "上","下","左","右"
        ]

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        batch_res = []
        for res in batch_preds:
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:6])
            work_idx = np.argmax(res[6:8])
            peng_idx = np.argmax(res[8:12])
            fx_idx = np.argmax(res[12:])
            if res[color_idx] >= self.color_threshold:
                color_info = f"{self.color_list[color_idx]}, prob: {res[color_idx]}"
            else:
                color_info = ""

            if res[work_idx + 6] >= self.work_threshold:
                work_info = f"{self.work_list[work_idx]}, prob: {res[work_idx + 6]}"
            else:
                work_info = ""

            if res[peng_idx + 8] >= self.peng_threshold:
                peng_info = f"{self.peng_list[peng_idx]}, prob: {res[peng_idx + 8]}"
            else:
                peng_info = ""

            if res[fx_idx + 12] >= self.fx_threshold:
                fx_info = f"{self.fx_list[fx_idx]}, prob: {res[fx_idx + 12]}"
            else:
                fx_info = ""

            label_res = {
                'color': color_info,
                "working": work_info,
                "status": peng_info, "position": fx_info}

            #
            # threshold_list = [self.color_threshold] * 6 + [self.work_threshold] * 2 + [self.peng_threshold] * 4 + [
            #     self.fx_threshold] * 4
            #
            # pred_res = (np.array(res) > np.array(threshold_list)
            #             ).astype(np.int8).tolist()
            batch_res.append(label_res)
        return batch_res
