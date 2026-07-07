# -*- coding: utf-8 -*-
#人脸
import os
import cv2
from time import sleep, localtime, strftime, time,strptime,mktime,time
from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
class Model(ModelClass):
    #获取当前时间
    def now_datetime(self): #
        return strftime('%Y-%m-%d %H:%M:%S', localtime(time()))
    def local_date(self): #年月日
        return strftime('%Y%m%d', localtime(time()))
    def local_datetime(self): #年月日时分秒秒
        return strftime('%Y%m%d%H%M%S', localtime(time()))

    def execute(self)->[list]:
        save_path = "./Picture_pkl"
        deviceid = self.cameraInfo.deviceId
        presetid = self.cameraInfo.presetId
        Flag = True
        eventype_path = os.path.join(save_path, self.logicModelName )
        deviceID_path = os.path.join(eventype_path, deviceid)  # deviceID路径
        presetid_path = os.path.join(deviceID_path, str(presetid))  # deviceID路径
        if not len(self.logicResult):
            # nyr = self.local_date()  # 年月日

            try:
                img_files = os.listdir(presetid_path)
            except:
                img_files = []
            now_time = self.local_datetime()  # 年月日时分秒秒
            picturexian_name = f'{deviceid}_{now_time}_{presetid}_alarm.jpeg'  # 报警图名字
            picturexian_path = os.path.join(presetid_path, picturexian_name)  # deviceID路径

            if len(img_files) == 0:
                if os.path.isdir(presetid_path) == False:
                    os.makedirs(presetid_path)
                # cv2.imshow("1", self.picture[0])
                # cv2.waitKey(0)
                cv2.imwrite(picturexian_path,self.picture)
                return [],[]
            else:
                for pic in img_files:
                    message = pic.split('_')
                start_time = f'{message[1][0:4]}:{message[1][4:6]}:{message[1][6:8]}:{message[1][8:10]}:{message[1][10:12]}:{message[1][12:14]}'
                now_time = strftime('%Y:%m:%d:%H:%M:%S', localtime())
                start_time1 =strptime(start_time, "%Y:%m:%d:%H:%M:%S")
                end_time1 = strptime(now_time, "%Y:%m:%d:%H:%M:%S")
                start_time2 = int(mktime(start_time1))
                end_time2 = int(mktime(end_time1))
                if_time = self.labelRules[self.logicModelName]["time"]
                # if_time = 1
                if int(end_time2 - start_time2) > int(if_time):
                    for pic in img_files:
                        if pic.endswith(".jpeg"):
                            pictureold_path = os.path.join(presetid_path, pic)  # deviceID路径
                            old_pic = cv2.imread(pictureold_path)
                            os.remove(pictureold_path)
                            cv2.imwrite(picturexian_path, self.picture)
                    # print(1)
                    # print(self.logicResult)
                    result_list = BoundingBox(0,1.0,int(100),int(101),int(100),int(101),999,999,self.logicModelName)
                    old_pics = []
                    old_pics.append(self.pic2base64(old_pic))
                    # time_client_json["Pretreatment"] = 0
                    # time_client_json["post_time"] = 0
                    # time_client_json["processing_time"] = 0
                    # time_client_json["showtime"] = time() - start
                    # time_client_json["analyseresult"] = result_list
                    return [result_list],old_pics
                else:
                    return [],[]

        else:
            if os.path.isdir(presetid_path) == False:
                return [],[]
            else:
                img_files = os.listdir(presetid_path)
                for pic in img_files:  # 遍历文件夹
                    if pic.endswith(".jpeg"):
                        os.remove(presetid_path + '/' + pic)
                return [],[]

    def __del__(self):
        """
        后处理  类注销前操作的 内容  或者 抛出 无法捕捉的异常报错  执行的内容
        :return:
        """
        pass


