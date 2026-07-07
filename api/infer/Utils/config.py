# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2022年10月31日 13:27:08
@packageName Triton
@className config
@version 1.0.0


@describe TODO
"""
import os
import json
import yaml
from api.infer.Utils.Singletons import Singleton
@Singleton
class Cfg():
    """
    配置文件热更新
        配置文件不存在时自动生成配置文件
        配置文件存在时 读取配置文件
    """
    root_path = os.path.dirname(os.path.dirname(__file__))

    triton_model = os.path.join(root_path,"Triton_model/weights")
    model_pipline = os.path.join(root_path,"Model_pipline")
    configs = {}          # 基础配置文件
    eventDict = {}        # 事件类型表
    logicModelDict = {}   # 逻辑标签字典
    unlogicModelDict = {} # 非逻辑标签字典
    deviceSatus = {}
    nexusDict = {}
    def __init__(self):
        """
        构造时 就 读取所有py

        """
        self.load_all()

    def load_all(self):
        self.load_configs()
        self.load_logicmodelcfg()
        self.load_unlogicmodelcfg()
        self.load_nexusai()
    def create_modelcfg(self):
        dict = {}
        try:
            for modeldir in os.listdir(os.path.join(self.root_path, "Triton_model/weights")):
                filename = os.path.join(os.path.join(self.root_path, "Triton_model/weights", modeldir, "config.json"))
                with open(filename) as f:
                    jsons = json.load(f)
                    try:
                        dict[jsons['name']] = jsons['classes']
                    except Exception as e:
                        print(e)
                        continue
            yaml.dump(dict, open(os.path.join(self.root_path,"Config","model.yaml", ),"w"))
        except:
            pass
        return dict

    def load_logicmodelcfg(self):
        try:
            logicmodel_dict = yaml.safe_load(open(os.path.join(self.root_path,"Config","model.yaml"),encoding='utf-8'))
            state = 200
            self.logicModelDict = logicmodel_dict

        except Exception as e:
            state = 0
        return self.logicModelDict,state

    def load_unlogicmodelcfg(self):
        unlogicmodel_dict = {}
        tritonModelPath = os.path.join(self.root_path,"Triton_model/weights")
        try:
            state = 200
            for model in os.listdir(tritonModelPath):
                with open(f'{tritonModelPath}/{model}/config.json', encoding='utf-8') as a:
                    triton_json = json.load(a)
                    classes = triton_json.get("classes",None)
                    unlogicmodel_dict[model] = {"input":triton_json['input'],
                                                "output":triton_json['output'],
                                                "classes":triton_json['classes'],
                                                "conf":triton_json['conf_thres'],
                                                "iou":triton_json['iou_thres']}
                    if classes is None:
                        continue
            self.unlogicModelDict = unlogicmodel_dict
        except Exception as e:
            state = 0
        return self.unlogicModelDict,state


    def load_configs(self):
        try:
            configs = yaml.safe_load(open(os.path.join(self.root_path, "Config", "config.yaml"), encoding='utf-8'))
            state = 200
            self.configs = configs
        except Exception as e:
            state = 0
        return self.configs, state

    def load_nexusai(self):
        try:
            configs = json.load(open(os.path.join(self.root_path, "config.json"), encoding='utf-8'))
            state = 200
            self.nexusDict = configs

        except Exception as e:
            print(e)
            state = 0
        return self.nexusDict, state
