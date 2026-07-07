import numpy as np
import math


class FacePoseEstimator4Points:
    def __init__(self):
        # 3D面部模型关键点的相对比例（基于平均人脸）
        # 用于计算角度的比例关系
        self.eye_distance = 6.5  # 双眼间距（厘米）
        self.eye_to_nose = 3.0  # 眼到鼻尖距离
        self.nose_to_chin = 8.0  # 鼻尖到下巴距离

    def estimate(self, landmarks):
        """
        估计人脸姿态

        参数:
            landmarks: 字典，包含4个关键点坐标:
                      'left_eye', 'right_eye', 'nose_tip', 'chin'

        返回:
            包含yaw, pitch, roll三个角度的字典
        """
        # 检查必要的关键点
        required_points = ['left_eye', 'right_eye', 'nose_tip', 'chin']
        for point in required_points:
            if point not in landmarks:
                raise ValueError(f"缺少必要的关键点: {point}")

        # 提取关键点坐标
        le = np.array(landmarks['left_eye'], dtype=np.float32)
        re = np.array(landmarks['right_eye'], dtype=np.float32)
        nt = np.array(landmarks['nose_tip'], dtype=np.float32)
        ch = np.array(landmarks['chin'], dtype=np.float32)

        # 计算滚转角 (roll) - 基于双眼连线的倾斜角度
        # 双眼连线与水平线的夹角
        eye_line = re - le
        roll = math.atan2(eye_line[1], eye_line[0]) * 180 / math.pi

        # 计算偏航角 (yaw) - 基于鼻尖相对于双眼中线的偏移
        # 计算双眼中点
        eye_mid = (le + re) / 2
        # 计算鼻尖到双眼中线的水平偏移
        nose_offset_x = nt[0] - eye_mid[0]
        # 双眼间距
        eye_dist_pixel = np.linalg.norm(eye_line)
        # 归一化偏移量（-1到1范围）
        norm_offset = nose_offset_x / (eye_dist_pixel / 2)
        # 映射到角度（假设最大偏移对应±45度）
        yaw = -norm_offset * 45  # 负号是为了符合常规坐标系定义

        # 计算俯仰角 (pitch) - 基于面部特征的垂直比例
        # 计算从双眼中点到鼻尖的距离
        eye_to_nose_pixel = np.linalg.norm(nt - eye_mid)
        # 计算从鼻尖到下巴的距离
        nose_to_chin_pixel = np.linalg.norm(ch - nt)
        # 计算比例
        ratio = nose_to_chin_pixel / (eye_to_nose_pixel + nose_to_chin_pixel)
        # 映射到角度（正面约0.6，抬头变小，低头变大）
        # 这是经验公式，可能需要根据实际数据调整
        pitch = (ratio - 0.6) * 150  # 映射到角度

        return {
            'yaw': yaw,  # 偏航角（左右转头）
            'pitch': pitch,  # 俯仰角（上下抬头）
            'roll': roll  # 滚转角（左右歪头）
        }


# 使用示例
if __name__ == "__main__":
    # 示例关键点（实际应用中应该由人脸检测算法提供）
    # 正面人脸示例
    front_face = {
        'left_eye': (55, 80),  # 左眼坐标
        'right_eye': (400, 80),  # 右眼坐标
        'nose_tip': (320, 99),  # 鼻尖坐标
        'chin': (320, 400)  # 下巴坐标
    }

    # 侧脸示例
    side_face = {
        'left_eye': (220, 200),  # 左眼坐标
        'right_eye': (380, 190),  # 右眼坐标
        'nose_tip': (300, 270),  # 鼻尖坐标
        'chin': (290, 390)  # 下巴坐标
    }

    # 低头示例
低头_face = {
    'left_eye': (240, 180),  # 左眼坐标
    'right_eye': (400, 180),  # 右眼坐标
    'nose_tip': (320, 270),  # 鼻尖坐标
    'chin': (320, 390)  # 下巴坐标
}

estimator = FacePoseEstimator4Points()

print("正面人脸姿态:")
angles = estimator.estimate(front_face)
print(f"偏航角 (yaw): {angles['yaw']:.2f}°")
print(f"俯仰角 (pitch): {angles['pitch']:.2f}°")
print(f"滚转角 (roll): {angles['roll']:.2f}°\n")

print("侧脸人脸姿态:")
angles = estimator.estimate(side_face)
print(f"偏航角 (yaw): {angles['yaw']:.2f}°")
print(f"俯仰角 (pitch): {angles['pitch']:.2f}°")
print(f"滚转角 (roll): {angles['roll']:.2f}°\n")

print("低头人脸姿态:")
angles = estimator.estimate(低头_face)
print(f"偏航角 (yaw): {angles['yaw']:.2f}°")
print(f"俯仰角 (pitch): {angles['pitch']:.2f}°")
print(f"滚转角 (roll): {angles['roll']:.2f}°")
