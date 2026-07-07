# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2023年12月07日 18:51:59
@packageName Triton
@className Pnp
@version 1.0.0
@describe TODO
"""
import cv2
import numpy as np
import math
from api.infer.Utils.coordTransform_py.coordTransform_utils import gcj02_to_bd09
from api.infer.Utils.coordTransform_py.coordTransform_utils import bd09_to_gcj02
from api.infer.Utils.coordTransform_py.coordTransform_utils import wgs84_to_gcj02
from api.infer.Utils.coordTransform_py.coordTransform_utils import gcj02_to_wgs84
from api.infer.Utils.coordTransform_py.coordTransform_utils import bd09_to_wgs84
from api.infer.Utils.coordTransform_py.coordTransform_utils import wgs84_to_bd09
import pyproj

import warnings
warnings.filterwarnings("ignore")
class Pnp2solv():
    def __init__(self):
        self.utm_proj = pyproj.Proj(proj='utm', zone=51, ellps='WGS84')
        self.wgs84_proj = pyproj.Proj(init='epsg:4326')
        self.fov = 178
        self.principal_point_x = math.tan(math.radians(self.fov / 2))
        self.principal_point_y = self.principal_point_x
        self.focal_length_x, self.focal_length_y, self.f = 1000000000, 1000000000, 1000000000
        self.camera_matrix = np.array([
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dtype = {
            'CAPTURE_SERVICE_HIKVISION_SDK':0,
            'CAPTURE_SERVICE_DAHUATECH_SDK':1,
        }
    def dataFormat(self, jwds, ptzs, flag):
        '''
        区分海康大华 p不同 的问题 矫正 p
        '''
        jwds = np.array(jwds,dtype=np.float32)
        ptzs = np.array(ptzs,dtype=np.float32)
        if self.dtype[flag]:
            ptzs[:, 0] = 3600 - ptzs[:, 0]
        ptzs = ptzs[:,:-1]
        return jwds,ptzs
    def ptz_touv(self, ptzs):
        uvs = []  # 已知的2D点（图像坐标系中）
        for ptz in ptzs:
            j_pan, j_tilt = float(ptz[0]) / 10, float(ptz[1]) / 10
            if ptz[1] >= 2700 and ptz[1] <= 3600:
                j_tilt = float(3600 - ptz[1]) / 10
            u = 1 / math.tan(math.radians(j_tilt)) * math.cos(math.radians(j_pan))
            v = 1 / math.tan(math.radians(j_tilt)) * math.sin(math.radians(j_pan))
            uvs.append([u, v, 1.0])
        image_points = np.array(uvs, dtype=np.float64)
        return image_points

    # uv 转 世界坐标系
    def uv_toworld(self, camera_matrix, image_points):
        return np.dot(camera_matrix, image_points.T).T[:, 0:2]

    def jwd_towordld(self, jwds):
        world_temp = []
        for jwd in jwds:
            bd09_longitude, bd09_latitude = jwd
            # 因为使用了 wgs84 51R 投影地带 做一下转化
            wgs84_longitude, wgs84_latitude = bd09_to_wgs84(bd09_longitude, bd09_latitude)
            # 输入经度和纬度坐标，然后将其转换为UTM坐标
            utm_easting, utm_northing = pyproj.transform(self.wgs84_proj, self.utm_proj, wgs84_longitude, wgs84_latitude)
            world_temp.append([utm_easting, utm_northing, 0.0])
            # print(f"UTM东坐标, UTM北坐标: {utm_easting},{utm_northing}")
        # 已知的3D点（世界坐标系中）
        world_points = np.array(world_temp, dtype=np.float64)
        return world_points

    def solvePNP(self, world_points, image_points, camera_matrix, ex=np.zeros((4, 1)), flag=0):
        success, vector_rotation, vector_translation = cv2.solvePnP(world_points, image_points, camera_matrix,ex, flags=flag)
        if success:
            return vector_rotation , vector_translation
        return None, None

    def jwd_topt(self,jwd,vector_translation ,rotationVector,vector_rotation_R,image_points,flag,ptzs):
        vector_rotationR_translation = np.hstack((vector_rotation_R, vector_translation))
        vector_rotationR_translation = np.vstack((vector_rotationR_translation, [0, 0, 0, 1]))
        new_bd09_longitude, new_bd09_latitude = jwd
        wgs84_longitude, wgs84_latitude = bd09_to_wgs84(new_bd09_longitude, new_bd09_latitude)
        # 输入经度和纬度坐标，然后将其转换为UTM坐标
        utm_easting, utm_northing = pyproj.transform(self.wgs84_proj, self.utm_proj, wgs84_longitude, wgs84_latitude)
        # 直接通过旋转平移函数进行图像坐标系的换算
        result_uv = np.dot(vector_rotationR_translation, np.array([utm_easting, utm_northing, 0., 1]).T)
        # 齐次转非齐次 有平移向量存在 干掉他！
        result_uv = result_uv[0:3] / result_uv[2]
        # print("result_uv by mic : ", result_uv)

        # 通过projectPoints函数求解图像坐标系坐标
        uv2d, jacobian = cv2.projectPoints(np.array([(utm_easting, utm_northing, 0.0)]),
                                           rotationVector, vector_translation,
                                           self.camera_matrix, np.zeros((4, 1)))

        u, v = uv2d[0][0]
        result = np.dot((np.matrix(self.camera_matrix)).I, np.array([[u], [v], [1]], np.float64))
        # print("result_uv by projectPoints :", result)
        # 由于象限的存在需要分两类讨论，此处可优化。
        type1_range = [[180, 270], [270, 360], [0, 90], [90, 180]]
        type1_delta = [180, 360, 0, 180]
        type2_range = [[0, 90], [90, 180], [180, 270], [270, 360]]
        type2_delta = [0, 180, 180, 360]

        # print((float)(ptzs[0][0])/10.0, image_points[0,0],image_points[0,1])
        # 判断 t的象限
        print(image_points)
        index = self.get_index(image_points[0, 0], image_points[0, 1])
        type_range = []
        type_delta = []
        if type1_range[index][0] <= (float)(ptzs[0][0]) / 10.0 < type1_range[index][1]:
            type_range = type1_range
            type_delta = type1_delta
        else:
            type_range = type2_range
            type_delta = type2_delta

        # 将 图像坐标系转换成p和t
        tan_j_pan = v / u
        j_delta = type_delta[self.get_index(u, v)]
        # 弧 转 度数
        j_pan = j_delta + math.degrees(math.atan(tan_j_pan))
        j_tilt = math.degrees(math.atan(v / math.sin(math.radians(j_pan)) / self.f))
        print(j_tilt)
        print(j_pan-j_delta)
        print(j_delta)
        if self.dtype[flag]:
            j_pan = 360 - j_pan
            if j_tilt >= 0:
                j_tilt = 360- j_tilt
            else:
                j_tilt = 90+ j_tilt
        else:
            if j_tilt >= 0:
                j_tilt = 90 - j_tilt
            else:
                j_tilt = 90 + j_tilt

        return j_pan*10,j_tilt*10
    def pt_tojwd(self,ptz,vector_translation,vector_rotation_R):
        vector_rotationR_translation = np.hstack((vector_rotation_R, vector_translation))
        vector_rotationR_translation = np.vstack((vector_rotationR_translation, [0, 0, 0, 1]))
        p,t,z =ptz
        if t >= 2700 and t <= 3600:
            t = float(3600 - t) / 10
        p, t = float(p) / 10, float(t) / 10
        u = (1 / math.tan(math.radians(t))) * math.cos(math.radians(p))
        v = (1 / math.tan(math.radians(t))) * math.sin(math.radians(p))
        # 需要通过utm是0的约束，求解出k，从而确定图像坐标系下的坐标
        vector_rotationR_translation_I = np.matrix(vector_rotationR_translation).I  # 先转矩阵 在求逆
        # 通过逆矩阵计算出 x，y
        line_third = vector_rotationR_translation_I[2]
        k = line_third[0, 3] / (line_third[0, 0] * u + line_third[0, 1] * v + line_third[0, 2])
        k = abs(k)
        # 将pt最终转换成图像坐标系
        camera_xyz = np.array([u * k, v * k, k, 1.0], np.float64)
        # 通过转换矩阵，将图像坐标系转换成 世界坐标系，即utm坐标系 逆运算
        result_utm = np.dot(vector_rotationR_translation_I, (camera_xyz).T)

        # utm坐标系转换成 wgs84坐标系再转换成bd09坐标系
        result_wgs84_longitude, result_wgs84_latitude = pyproj.transform(self.utm_proj, self.wgs84_proj, result_utm[0, 0],
                                                                         result_utm[0, 1])
        result_bd09_longitude, result_bd09_latitude = wgs84_to_bd09(result_wgs84_longitude, result_wgs84_latitude)
        return result_bd09_longitude, result_bd09_latitude

    def get_index(self, x, y):
        if x >= 0 and y >= 0:  # ++ 1象限
            return 0
        if x < 0 and y >= 0:  # -+ 2 象限
            return 1
        if x < 0 and y < 0:  # --3 象限
            return 2
        if x >= 0 and y < 0:  # +- 4象限
            return 3
pnp = Pnp2solv()
# if __name__ == '__main__':
#     pnp = Pnp2solv()
#
#     # 获取旋转矩阵
#     jwds = [
#         [121.88558, 29.49093],
#         [121.88621, 29.48779],
#         [121.88335, 29.48817],
#         [121.88318, 29.49465]
#     ]
#     # 标定ptz
#     ptzs = [
#         [1916, 3403],
#         [2573, 3476],
#         [3035, 3405],
#         [1159, 3514]
#     ]
#     # 接口格式化
#     jwd, ptz = pnp.dataFormat(jwds,ptzs,"CAPTURE_SERVICE_DAHUATECH_SDK")
#     # pt 转平面
#     images_points = pnp.ptz_touv(ptzs)
#     # 平面 转世界坐标系
#     images_points = pnp.uv_toworld(pnp.camera_matrix,images_points)
#     print(images_points)
#     # 经纬度转世界坐标系
#     world_points = pnp.jwd_towordld(jwds)
#     vector_rotation, vector_translation = pnp.solvePNP(world_points,images_points,pnp.camera_matrix)
#     print(vector_rotation, vector_translation)
#
#     vector_rotation_R, _ = cv2.Rodrigues(vector_rotation)  # 返回旋转矩阵 和旋转向量
#     # 输入
#     p,t,z = 2538, 69,1
#     pnp.pt_tojwd((p,t,z),vector_translation,vector_rotation_R)
#     ac = pnp.jwd_topt((121.88558, 29.49093),vector_translation,vector_rotation,vector_rotation_R,images_points,"CAPTURE_SERVICE_DAHUATECH_SDK",ptzs)
#     print(ac)