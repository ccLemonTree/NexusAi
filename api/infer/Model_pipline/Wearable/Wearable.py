# -*- coding: utf-8 -*-
from copy import copy

from api.infer.Utils.class_info import ModelClass
from api.infer.running import analyseRun
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


# Heights are relative to the top of the detected face.
_FACE_ZONES = {
    "Wearable-mask": (0.50, 1.00),
    "Wearable-sunglasses": (0.15, 0.60),
    "Wearable-hat": (-0.25, 0.35),
}


def _matching_face_index(item, crop_origin, faces, rule_name):
    if item.width() <= 0 or item.height() <= 0:
        return None

    lower, upper = _FACE_ZONES[rule_name]
    is_hat = rule_name == "Wearable-hat"
    left, top = crop_origin
    x, y = item.center_absolute()
    x += left
    y = top + (item.y2 if is_hat else y)
    candidates = []
    for face_index, face in faces.items():
        if not face.x1 <= x <= face.x2 or (is_hat and item.y1 + top > face.y1):
            continue
        height = (y - face.y1) / face.height()
        if lower <= height <= upper:
            center_x, _ = face.center_absolute()
            distance = abs(x - center_x) / face.width() + abs(height - (lower + upper) / 2)
            candidates.append((distance, face_index))
    return min(candidates)[1] if candidates else None


def run_wearable_rules(image, detections, camera_info, model_config, label_rules):
    try:
        face_label, = model_config[next(iter(label_rules))][0]["label"]
        item_to_rule = {}
        item_rules = {}
        for rule_name, params in label_rules.items():
            config = model_config[rule_name]
            settings = {
                "conf": params.get("conf", config["param"][0]["conf"]),
                "iou": params.get("iou", config["param"][1]["iou"]),
            }
            for item_label in config[1]["label"]:
                item_to_rule[item_label] = rule_name
                item_rules[item_label] = settings

        faces = [box for box in detections
                 if box.classname == face_label and box.width() > 0 and box.height() > 0]
        faces_by_item = {
            label: {index: face for index, face in enumerate(faces)
                    if face.confidence > settings["conf"]}
            for label, settings in item_rules.items()
        }
        image_height, image_width = image.shape[:2]
        results = {}
        for face_index, face in enumerate(faces):
            crop_rules = {label: settings for label, settings in item_rules.items()
                          if face_index in faces_by_item[label]}
            if not crop_rules:
                continue

            # Leave room for hats that extend above and to either side of the face.
            padding = face.width() // 2
            left = max(0, face.x1 - padding)
            right = min(image_width, face.x2 + padding)
            top = max(0, face.y1 - face.height())
            bottom = min(image_height, face.y2 + face.height() // 4)
            crop = image[top:bottom, left:right]
            if not crop.size:
                continue

            items = analyseRun(list(crop_rules), [crop, crop], camera_info, label_rules=crop_rules)
            matched_rules = set()
            for item in items:
                settings = crop_rules.get(item.classname)
                if settings is None or item.confidence <= settings["conf"]:
                    continue
                rule_name = item_to_rule[item.classname]
                if rule_name in matched_rules:
                    continue
                candidate_faces = faces_by_item[item.classname]
                if _matching_face_index(item, (left, top), candidate_faces, rule_name) != face_index:
                    continue

                matched_rules.add(rule_name)
                box = copy(face)
                box.classname = rule_name
                results.setdefault(rule_name, {"bbox": [], "imglist": []})["bbox"].append(box)
        return results
    except Exception:
        logger.exception("Wearable recognition failed")
        return {}


class Model(ModelClass):
    def execute(self):
        results = run_wearable_rules(
            self.picture, self.logicResult, self.cameraInfo, self.cfg.logicModelDict,
            {self.logicModelName: {"conf": self.conf, "iou": self.iou}},
        )
        return results.get(self.logicModelName, {}).get("bbox", []), []
