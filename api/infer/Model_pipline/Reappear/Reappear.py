# -*- coding: utf-8 -*-
import json
from time import time

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.logic_state import atomic_write, locked_state


class Model(ModelClass):
    def execute(self):
        with locked_state(self.cameraInfo, self.logicModelName) as directory:
            state_path = directory / "absence.json"
            if self.logicResult:
                state_path.unlink(missing_ok=True)
                return [], []

            now = int(time())
            try:
                with state_path.open(encoding="utf-8") as source:
                    state = json.load(source)
            except FileNotFoundError:
                state = None

            if state is not None:
                if now - state["started_at"] <= int(self.labelRules[self.logicModelName]["time"]):
                    return [], []

            current = {"started_at": now, "image": self.pic2base64(self.picture)}
            atomic_write(state_path, json.dumps(current).encode("utf-8"))
            if state is None:
                return [], []

            result = BoundingBox(0, 1.0, 100, 101, 100, 101, 999, 999, self.logicModelName)
            return [result], [state["image"]]
