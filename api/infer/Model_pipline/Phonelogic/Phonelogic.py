# -*- coding: utf-8 -*-
import base64
import json
import math
import time
from copy import copy
from uuid import uuid4

import cv2
from redis.exceptions import WatchError

from api.infer.Utils.boundingbox import BoundingBox
from api.infer.Utils.class_info import ModelClass
from api.infer.running import analyseRun
from tools.api import calculate_iou
from tools.init import get_redis_client
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from tools.logic_state import state_key


class Model(ModelClass):
    def execute(self):
        try:
            if not self.cameraInfo.deviceId:
                raise ValueError("Phonelogic requires deviceId")

            config = self.cfg.logicModelDict[self.logicModelName]
            rule = {key: value for item in config["param"] for key, value in item.items() if key != "desc"}
            rule.update(self.labelRules.get(self.logicModelName, {}))
            duration = float(rule["analysis_time"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("analysis_time must be a positive number of seconds")
            rule.update(analysis_time=duration, conf=self.conf, iou=self.iou)

            now = time.time()
            people = [box for box in self.logicResult if box.classname in config[0]["label"]]
            phone_labels = config[1]["label"]
            using_phone = []
            for person in people:
                crop = self.picture[person.y1:person.y2, person.x1:person.x2]
                detections = analyseRun(phone_labels, [crop, crop], self.cameraInfo, box_info=person) if crop.size else []
                using_phone.append(any(box.classname in phone_labels for box in detections))

            key = state_key(self.cameraInfo, self.logicModelName, rule)
            return self._update_cache(f"phonelogic:{{{key}}}", now, duration, people, using_phone)
        except Exception:
            logger.exception("%s failed to analyse phone usage", self.logicModelName)
            return [], []

    def _update_cache(self, key_prefix, now, duration, people, using_phone):
        client = get_redis_client()
        state_key_name = f"{key_prefix}:state"
        ttl = max(3600, math.ceil(duration) + 60)
        for attempt in range(8):
            try:
                with client.pipeline() as pipeline:
                    pipeline.watch(state_key_name)
                    cached_state = pipeline.get(state_key_name)
                    state = json.loads(cached_state) if cached_state else None
                    if state and now <= state["observed_at"]:
                        return [], []

                    previous_tracks = state["tracks"] if state else []
                    pairs = [
                        (calculate_iou(person.box(), track["last_box"]), person_index, track_index)
                        for person_index, person in enumerate(people)
                        for track_index, track in enumerate(previous_tracks)
                    ]
                    matches = {}
                    matched_tracks = set()
                    for overlap, person_index, track_index in sorted(pairs, reverse=True):
                        if overlap <= 0.3:
                            break
                        if person_index not in matches and track_index not in matched_tracks:
                            matches[person_index] = previous_tracks[track_index]
                            matched_tracks.add(track_index)

                    tracks = []
                    boxes, images = [], []
                    start_image = None
                    jpeg = None
                    for index, (person, positive) in enumerate(zip(people, using_phone)):
                        if not positive:
                            continue
                        track = matches.get(index)
                        if track is not None and now - track["started_at"] >= duration:
                            evidence = pipeline.get(track["image"])
                            if evidence is not None:
                                metadata = {"eventId": track["event_id"]}
                                first = BoundingBox(
                                    *track["start_box"], self.logicModelName,
                                    parames_vector={**metadata, "threeDtimes": 0},
                                )
                                current = copy(person)
                                current.classname = self.logicModelName
                                current.threeDtimes = 1
                                current.parames_vector = {**person.parames_vector, **metadata, "threeDtimes": 1}
                                boxes.extend((first, current))
                                images.append(base64.b64encode(evidence).decode("ascii"))
                                # Rearm on the next positive frame.
                                continue
                            track = None

                        if track is None:
                            if start_image is None:
                                success, encoded = cv2.imencode(".jpg", self.picture)
                                if not success:
                                    raise ValueError("Failed to encode phone evidence")
                                start_image = f"{key_prefix}:image:{uuid4().hex}"
                                jpeg = encoded.tobytes()
                            track = {
                                "event_id": uuid4().hex,
                                "started_at": now,
                                "start_box": [person.classID, person.confidence, person.x1, person.x2,
                                              person.y1, person.y2, person.image_width, person.image_height],
                                "image": start_image,
                            }
                        track["last_box"] = person.box()
                        tracks.append(track)

                    # Consume the event and release its evidence in the same transaction.
                    pipeline.multi()
                    pipeline.set(state_key_name, json.dumps({"observed_at": now, "tracks": tracks}), ex=ttl)
                    retained = {track["image"] for track in tracks}
                    for image in retained:
                        if image == start_image:
                            pipeline.set(image, jpeg, ex=ttl)
                        else:
                            pipeline.expire(image, ttl)
                    obsolete = {track["image"] for track in previous_tracks} - retained
                    if obsolete:
                        pipeline.delete(*obsolete)
                    pipeline.execute()
                    return boxes, images
            except WatchError:
                if attempt == 7:
                    raise
