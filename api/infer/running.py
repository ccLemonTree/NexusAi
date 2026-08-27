
from tools.init import  tritonServer
from api.infer.Utils.result_utils import *
from tools.concurrency import get_inference_executor

executor = get_inference_executor()
from tools.init import cfg
from api.infer.Utils.analyse_utils import LabelToModel
from api.infer.Utils.boundingbox import BoundingBox


def expand_logic_label_rules(label_rules, logic_model_dict):
    expanded_rules = dict(label_rules)
    for logic_label, rule in label_rules.items():
        for stage in logic_model_dict.get(logic_label, {}).values():
            if not isinstance(stage, dict):
                continue
            for label in stage.get("label", []):
                expanded_rules[label] = rule
    return expanded_rules


def analyseRun(setsLabel, imgs, camerInfo=CameraInfo(),label_rules={},roi=None, label_to_detect=[], box_info=BoundingBox(0, 0, 0, 0, 0, 0, 1, 1, "cls")):
    """
    param:
    setsLabel : 逻辑（逻辑标签不嵌套） + 非逻辑  [car, illegal-car,...]
    imgs : 双份图片
    camerInfo : 设备信息
    label_to_detect ： 传入的分析标签
    box_info: 传入的boundingbox
    Return : [BoundingBox , ...]
    """
    img = imgs[0]
    label_rules = expand_logic_label_rules(label_rules, cfg.logicModelDict)
    # 列表展平 去重
    setsLabel = set(setsLabel)  # ['car', 'illegal-car']
    # 逻辑标签集合
    logicLabels = cfg.logicModelDict.keys()
    # 首次分析 的标签集合 >>> {'car', 'truck'}
    unlogicAnalysis = set()
    logicAnalysis = set()
    # 提取 首次逻辑+非逻辑 分析的标签
    for lab in setsLabel:
        if lab in logicLabels:
            for i in cfg.logicModelDict[lab][0]["label"]:
                logicAnalysis.add(i)
        else:
            unlogicAnalysis.add(lab)
    # 结果并集 逻辑首次 + 非逻辑
    firstAnalysis = unlogicAnalysis | logicAnalysis
    # 标签找模型
    analyseModels,modelConf = LabelToModel(firstAnalysis, cfg.unlogicModelDict)
    for key,value in modelConf.items():
        modelConf[key] = label_rules.get(key,value)


    firstResult, unlogicAnalyseTime = Unlogic_run(img, analyseModels, tritonServer, executor,label_rules= modelConf,box_info=box_info)
    boxes = firstResult

    # 存储 逻辑任务的线程
    fuctureList = []
    logicAnalysisDict = {}
    # 循环所有ROI 结构体

    for lab in setsLabel:
        # 逻辑标签首次需要的标签结果
        logicFirstLabel = []
        # 逻辑标签标志符 用于判断是否为逻辑标签进行 逻辑块分析
        logicFlag = True if lab in logicLabels else False
        comparisonList = cfg.logicModelDict[lab][0]["label"] if logicFlag else [lab]
        for bounding in firstResult:
            if bounding.classname in comparisonList:
                if logicFlag:
                    logicFirstLabel.append(bounding)
        if logicFlag:
            logicLabList = logicAnalysisDict.get(lab, [])
            logicLabList += logicFirstLabel
            logicAnalysisDict[lab] = logicLabList

    for lab, logicList in logicAnalysisDict.items():
        fuctureList.append(executor.submit(
            logic_run, imgs, logicList, camerInfo, lab, tritonServer, label_rules))  # 提交任务

    # 获取逻辑标签结果
    logicResults = []
    for fucture in fuctureList:
        result = fucture.result()
        if len(list(result.values())[0]['bbox']) == 0:
            continue
        logicResults.append(result)
    for logic_result in logicResults:
        for key,value in logic_result.items():
            boxes = firstResult + value['bbox']
    return boxes

