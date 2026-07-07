from triton_client import triton_inference
import cv2
import numpy as np
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    dir = r"./Triton_model/weights"
    url = '192.168.1.25:40008'
    ti = triton_inference(dir,url)
    # 测试 yolov7
    # seetaface_post5net2
    # seetaface_post5net1
    # seetaface_det
    # seetaface_rec
    service_name = "retina_facetrt"
    images_path = r"C:\Users\18088\Pictures\2.jpeg"
    return_labels = ti.detected_labels(service_name)
    result_to_return = ti.run(service_name, images_path,{'filename':'111'})
    for i in result_to_return:
       print(i)
    cv2.imshow("!",images_path)
    cv2.waitKey(0)
    # for i in result_to_return[0]:
    #     cv2.rectangle(images_path,(i.x1,i.y1),(i.x2,i.y2),(255,0,0),3,1)
    # print(images_path)
    # cv2.imshow("1",images_path)
    # cv2.waitKey(0)
    # feature, boxes = result_to_return
    # print(feature)
    # print(boxes)
    # print(result_to_return)


    # label_to_detect = []
    #
    # img = cv2.imread(images_path)
    # inputs = []
    # outputs = []
    # output_data = []
    # model_version = ""
    # conf_thres = 0.5
    # iou_thres = 0.5
    # INPUT_DATA = ["_input_123"]
    # OUTPUT_DATA = ["fc1_act_50"]
    # label_names = []
    # input_shape = []
    # # cv2.imshow("1",img)
    # # cv2.waitKey(0)
    # for input in INPUT_DATA:
    #     inputs.append(grpcclient.InferInput(input, [1,3,248,248], "FP32"))
    # for OutputName in OUTPUT_DATA:
    #     outputs.append(grpcclient.InferRequestedOutput(OutputName))
    # # input_image_buffer = preprocess(img, input_shape)
    # # input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    #
    # resize_img, ratio, _ = letterbox(img, (248, 248), auto=False, scaleFill=True)
    # img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # img = img - np.array([123, 117, 104])
    # img_shape = img.shape
    # img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    # input_image_buffer = img.transpose((0, 3, 1, 2)).astype(np.float32)
    # input_image = input_image_buffer
    # print(input_image.shape)
    # inputs[0].set_data_from_numpy(input_image)
    #
    #
    # results = triton_client.infer(model_name=service_name,
    #                               inputs=inputs,
    #                               outputs=outputs,
    #                               model_version=model_version,
    #                               client_timeout=30000)
    # for output in OUTPUT_DATA:
    #     results.as_numpy("fc1_act_50")
    # for obj in OUTPUT_DATA:
    #     output_data.append(results.as_numpy("fc1_act_50"))
    # print(output_data)


