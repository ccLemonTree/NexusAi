# -*- coding: utf-8 -*-
from copy import copy

from api.infer.Utils.class_info import ModelClass


class Model(ModelClass):
    def execute(self):
        labels = self.cfg.logicModelDict[self.logicModelName][0]["label"]
        people = [box for box in self.logicResult if box.classname in labels]
        count = len(people)
        alarm = {
            "Intrusion-at-least-two": count >= 2,
            "Intrusion-any": count >= 1,
            "Intrusion-except-two": count > 0 and count != 2,
            "Intrusion-exactly-one": count == 1,
        }[self.logicModelName]
        if not alarm:
            return [], []

        boxes = [copy(person) for person in people]
        for box in boxes:
            box.classname = self.logicModelName
        return boxes, []
