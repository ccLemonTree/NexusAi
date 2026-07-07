import time
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.resnest.utils.processing import *
import numpy as np
# import torch

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }
def resize(img, size, interpolation='BILINEAR'):
    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])

def center_crop_numpy(img, crop_size):
    """
    对图像进行中心裁剪的 NumPy 实现。

    参数:
    - img: 输入的图像，可以是 NumPy 数组。
    - crop_size: 想要裁剪的目标尺寸，可以是单个整数或元组 (height, width)。

    返回:
    - 裁剪后的图像。
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    height, width = img.shape[:2]
    crop_height, crop_width = crop_size

    start_h = (height - crop_height) // 2
    start_w = (width - crop_width) // 2

    cropped_img = img[start_h:start_h+crop_height, start_w:start_w+crop_width, ...]

    return cropped_img
def softmax( f ):
    # 坏的实现: 数值问题
    # return np.exp(f) / np.sum(np.exp(f))
    return 1 / (1 + np.exp(-f))

def to_tensor(pic):
    pic = np.transpose(pic,(2, 0, 1))

    # pic = torch.from_numpy(pic)
    return pic
    # return pic.float().div(255)


def preprocess(img, dst_width=224, dst_height=224):
    imh, imw = img.shape[:2]
    m = min(imh, imw)
    top, left = (imh - m) // 2, (imw - m) // 2
    img_pre = img[top:top + m, left:left + m]
    img_pre = cv2.resize(img_pre, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)

    img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)

    return img_pre


def resnet(triton_client, service_name, init_data, img,label_rules,box_info):
    pretreatment_start = time.time()  # 时间测试
    time_json = {
        "filename": "",
        "model": "",
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }
    time_json["model"] = service_name

    model_version = ""
    INPUT_DATA = []
    OUTPUT_DATA = []
    label_names = []
    input_shape = []
    if not bool(init_data):
        print("模型信息没有读到")
    else:
        for key, value in init_data.items():
            if key == service_name:
                INPUT_DATA = value["input"]
                OUTPUT_DATA = value["output"]
                model_version = value["model_version"]
                label_names = value["label_names"]
                for input in INPUT_DATA:
                    input_shape.append(input["dims"][-2])
                    input_shape.append(input["dims"][-1])
    inputs = []
    outputs = []
    det_obj = []
    IMAGES = img

    # import cv2
    # cv2.imshow('1',img)
    # IMAGES1 = IMAGES/255
    # cv2.imshow("2",IMAGES1)
    # cv2.waitKey(0)
    # IMAGES = to_tensor(IMAGES)
    # IMAGES = resize(IMAGES, input_shape, interpolation='BILINEAR')
    # IMAGES = center_crop_numpy(IMAGES,input_shape)
    # mean = np.array([0,0,0])
    # std = np.array([1,1,1])
    # # IMAGES = IMAGES / 255
    # IMAGES = (IMAGES - mean) / std
    # # IMAGES = cv2.resize(IMAGES,input_shape)
    # IMAGES = center_crop_numpy(IMAGES,input_shape)
    #
    # IMAGES = to_tensor(IMAGES)
    IMAGES = preprocess(IMAGES,input_shape[0],input_shape[1])
    IMAGES = [np.array([IMAGES])]

    for input, input_image in zip(INPUT_DATA, IMAGES):
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))

    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))

    input_image = IMAGES[0].astype(np.float32)

    inputs[0].set_data_from_numpy(input_image)

    pretreatment_end = time.time()  # 时间测试
    # 检测模型服务是否正常 Ready
    if triton_client.is_model_ready(service_name, model_version):
        results = triton_client.infer(model_name=service_name,
                                      inputs=inputs,
                                      outputs=outputs,
                                      model_version=model_version,
                                      client_timeout=30000)
        for output in OUTPUT_DATA:
            results.as_numpy(output["name"])
        for obj in OUTPUT_DATA:
            det_obj.append(results.as_numpy(obj["name"]))
    postprocessing_start = time.time()  # 时间测试
    detects = list(label_rules.keys())
    # print(det_obj)
    for zz in det_obj:
        onnx_confidence = softmax(zz)
        zz = onnx_confidence
        onnx_index = np.argmax(zz, axis=1)
        confidence =zz[0][onnx_index]
        result_to_return = []
        if label_names[onnx_index[0]] in detects:
            if box.confidence > (label_rules[label_name]['conf'] if isinstance(label_rules,dict) else conf_thres):
                result_to_return.append(BoundingBox(onnx_index[0], float(confidence)/10, 0,1,0,1, img.shape[1], img.shape[0],label_names[onnx_index[0]]))
    return result_to_return, time_json
