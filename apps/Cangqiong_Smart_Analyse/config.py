import requests
from fastapi import APIRouter
import os
from tools.init import client, chat_infer, cfg
from api.infer.Utils.analyse_utils import LabelToModel
import json

router = APIRouter()


# 模型能力查询 （提供当前支持的所有模型）并提供在线状态，不在线需要
@router.post("/abilityProvided")
async def abilityProvided():
    """


    """

    onlines_label = []

    # 获取Triton加载的模型状态
    online = []
    response = requests.post(f"http://{os.getenv('TRITON_MODEL_STATUS')}/v2/repository/index")
    model_status = response.json()
    for i in model_status:
        status = i.get("state", "offline")
        if status == "READY":
            online.append(i['name'])

    # 逻辑标签取决于基础标签所属的模型是否存在，因此先判断基础标签

    logic_config = cfg.logicModelDict
    for key, value in logic_config.items():
        belongmodel = set()
        flag = True
        for skey, svalue in logic_config[key].items():
            if isinstance(svalue, dict):
                data = svalue.get('label', None)
                if data is not None:
                    for d in data:
                        isdata,_ = LabelToModel([d], cfg.unlogicModelDict)
                        if len(isdata) == 0:
                            flag = False
                            break
                        else:
                            belongmodel.add(*isdata.keys())

        if logic_config[key]['show']:
            onlines_label.append({
                key: {"status": "online" if flag else "offline",
                      "params": logic_config[key]['param'],
                      "type": "logic",
                      "belong_model": list(belongmodel),
                      "desc": logic_config[key]['label_desc']
                      }
            })
    for small_modle in os.listdir("api/infer/Triton_model/weights"):
        status = "offline"
        if small_modle in online:
            status = "online"
        with open(os.path.join("api/infer/Triton_model/weights", small_modle, "config.json"),encoding='utf-8') as f:
            data = json.load(f)
            if str(data.get('show', True)).lower() == "false" or data.get('show') is False:
                continue
            labels = data['classes']
            labels_zh = data['classes_zh']
            for index, name in enumerate(labels):
                try:
                    onlines_label.append(
                        {
                            name: {
                                "status": status,
                                "params": [{
                                        "conf": 0.65,
                                        "desc": "置信度"
                                    },
                                    {
                                        "iou": 0.2,
                                        "desc": "重叠度"
                                    }
                                ],
                                "belong_model": [small_modle],
                                "desc": labels_zh[index],
                                "type": "unlogic"
                            }
                        }
                    )
                except Exception:
                    pass

    return onlines_label
