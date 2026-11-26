# -*- coding: utf-8 -*-
import json
from copy import copy
from time import time

from redis.exceptions import WatchError

from api.infer.Utils.class_info import ModelClass
from tools.init import get_redis_client
from tools.logic_state import state_key


class Model(ModelClass):
    def execute(self):
        if not self.cameraInfo.deviceId:
            raise ValueError("Loitering requires deviceId")

        config = self.cfg.logicModelDict[self.logicModelName]
        rule = {key: value for param in config["param"] for key, value in param.items() if key != "desc"}
        rule.update(self.labelRules.get(self.logicModelName, {}))
        duration = int(rule["time"])
        if duration <= 0:
            raise ValueError("Loitering time must be positive seconds")
        rule.update(time=duration, conf=self.conf, iou=self.iou)

        people = [box for box in self.logicResult if box.classname in config[0]["label"]]
        now = time()
        scope = state_key(self.cameraInfo, self.logicModelName, rule)
        key = f"loitering:{scope}:state"
        ttl = max(3600, duration + 60)

        client = get_redis_client()
        for attempt in range(8):
            try:
                with client.pipeline() as pipeline:
                    pipeline.watch(key)
                    data = pipeline.get(key)
                    state = json.loads(data) if data else None
                    if state and now <= state["observed_at"]:
                        return [], []

                    started_at = state["started_at"] if state else None
                    alarm = bool(people) and started_at is not None and now - started_at >= duration
                    if not people or alarm:
                        started_at = None
                    elif started_at is None:
                        started_at = now

                    pipeline.multi()
                    pipeline.set(key, json.dumps({"observed_at": now, "started_at": started_at}), ex=ttl)
                    pipeline.execute()
                    break
            except WatchError:
                if attempt == 7:
                    raise

        if not alarm:
            return [], []
        boxes = [copy(person) for person in people]
        for box in boxes:
            box.classname = self.logicModelName
        return boxes, []
