import cv2
import numpy as np
from api.infer.Utils.boundingbox import BoundingBox


def scale_target_landmarks(original_h, original_w, target_h, target_w):
    """缩放标准目标关键点到目标图像尺寸"""
    original_target = {
        "left_eye": [299, 436],
        "right_eye": [458, 443],
        "nose": [378, 534],
        "left_mouth_corner": [309, 605],
        "right_mouth_corner": [434, 610]
    }
    return {
        key: [x * (target_w / original_w), y * (target_h / original_h)]
        for key, (x, y) in original_target.items()
    }


def transformation_from_points(points1, points2):
    """计算两个点集之间的仿射变换矩阵（使用SVD）"""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, _, Vt = np.linalg.svd(points1.T @ points2)
    R = (U @ Vt).T

    scale = s2 / s1
    translation = c2 - scale * (R @ c1.T).T

    return np.array([
        [scale * R[0, 0], scale * R[0, 1], translation[0]],
        [scale * R[1, 0], scale * R[1, 1], translation[1]]
    ])


def apply_affine_transform(points, M):
    """对点集应用仿射变换"""
    ones = np.ones(shape=(len(points), 1))
    points_homogeneous = np.hstack([points, ones])
    return (M @ points_homogeneous.T).T


def align_face_and_map_coordinates(img, bbox_obj, target_size=None):
    """
    对人脸图像进行对齐、裁剪和坐标映射

    参数:
        img: 输入图像 (BGR)
        bbox_obj: BoundingBox对象，包含人脸框和关键点
        target_size: (可选) 矫正后图像的尺寸 (h, w)

    返回:
        cropped_face: 裁剪后的人脸图像
        cropped_bbox: 裁剪后的人脸框 (x1, y1, x2, y2)
        cropped_landmarks: 裁剪后的人脸关键点
        angle: 旋转角度
    """
    try:
        h, w = img.shape[:2]

        # 1. 提取人脸框和关键点
        bbox = [bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2]
        landmarks = bbox_obj.parames_vector

        required_keys = ["left_eye", "right_eye", "nose", "left_mouth_corner", "right_mouth_corner"]
        missing_keys = [key for key in required_keys if key not in landmarks]
        if missing_keys:
            raise ValueError(f"缺少关键点: {missing_keys}")

        # 2. 准备标准关键点
        key_order = required_keys
        target_landmarks = scale_target_landmarks(1050, 750, h, w)
        src_points = np.float64([landmarks[key] for key in key_order])
        dst_points = np.float64([target_landmarks[key] for key in key_order])

        # 3. 计算变换矩阵
        M = transformation_from_points(src_points, dst_points)
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        # 4. 应用变换
        aligned_img = cv2.warpAffine(img, M[:2], (w, h))

        # 5. 变换边界框
        bbox_points = np.float64([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[2], bbox[1]]
        ])
        aligned_bbox_points = apply_affine_transform(bbox_points, M)
        aligned_bbox = [
            np.min(aligned_bbox_points[:, 0]),
            np.min(aligned_bbox_points[:, 1]),
            np.max(aligned_bbox_points[:, 0]),
            np.max(aligned_bbox_points[:, 1])
        ]

        # 6. 变换关键点
        aligned_landmarks = {}
        for key, point in landmarks.items():
            pt = np.float64([[point[0], point[1]]])
            transformed = apply_affine_transform(pt, M)[0]
            aligned_landmarks[key] = [float(transformed[0]), float(transformed[1])]

        # 7. 图像缩放（可选）
        if target_size:
            new_h, new_w = target_size
            scale_x, scale_y = new_w / w, new_h / h
            aligned_img = cv2.resize(aligned_img, (new_w, new_h))
            aligned_bbox = [x * scale_x if i % 2 == 0 else x * scale_y for i, x in enumerate(aligned_bbox)]
            for key in aligned_landmarks:
                aligned_landmarks[key] = [
                    aligned_landmarks[key][0] * scale_x,
                    aligned_landmarks[key][1] * scale_y
                ]
            h, w = new_h, new_w

        # 8. 裁剪人脸
        x1 = max(0, int(round(aligned_bbox[0])))
        y1 = max(0, int(round(aligned_bbox[1])))
        x2 = min(w - 1, int(round(aligned_bbox[2])))
        y2 = min(h - 1, int(round(aligned_bbox[3])))

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"裁剪区域无效: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        cropped_face = aligned_img[y1:y2+1, x1:x2+1]

        # 9. 裁剪后坐标
        cropped_bbox = [0, 0, x2 - x1, y2 - y1]
        cropped_landmarks = {
            key: [x - x1, y - y1] for key, (x, y) in aligned_landmarks.items()
        }

        # return cropped_face, cropped_bbox, cropped_landmarks, angle
        return cropped_face

    except Exception as e:
        print(f"人脸对齐失败: {e}")
        raise


