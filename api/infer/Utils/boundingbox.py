import numpy as np


class BoundingBox(object):
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height, classname, parames_vector={},
                 threeDtimes=0,father_info =None, mask=None):
        """
        数据结构类
        :param classID:        标签索引 0 ,1 ...
        :param confidence:     目标概率
        :param x1:             x1 坐标
        :param x2:             x2 坐标
        :param y1:             y1 坐标
        :param y2:             y2坐标
        :param image_width:    图片宽
        :param image_height:   图片高
        :param classname:      标签名称 person  dog ...
        :param parames_vector: 参数兼容
        """
        self.classID = int(classID)
        self.confidence = round(float(confidence), 2)
        self.image_width = image_width  # 9999 if image_width ==0 else
        self.image_height = image_height  # 9999 if image_height ==0 else
        self.orix1 = x1
        self.orix2 = x2
        self.oriy1 = y1
        self.oriy2 = y2
        self.x1 = int(np.clip(x1, 0, self.image_width))
        self.x2 = int(np.clip(x2, 0, self.image_width))
        self.y1 = int(np.clip(y1, 0, self.image_height))
        self.y2 = int(np.clip(y2, 0, self.image_height))
        self.threeDtimes = threeDtimes
        self.u1 = x1 / self.image_width
        self.u2 = x2 / self.image_width
        self.v1 = y1 / self.image_height
        self.v2 = y2 / self.image_height
        self.parames_vector = parames_vector
        self.classname = classname
        self.orig_classname = self.classname
        self.mask = mask

    def topLeft(self):
        return self.x1, self.y1

    def bottomRight(self):
        return self.x2, self.y2

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def center_boxs(self):
        return self.x1 + self.width() // 2, self.height() // 2 + self.y1

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def obj(self):
        return (self.x1, self.y1, self.width(), self.height(), self.confidence, self.classID)

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)

    def yolo_txt(self):
        cx = 0.5 * (self.u1 + self.u2)
        cy = 0.5 * (self.v1 + self.v2)
        cw = self.width() / self.image_width
        ch = self.height() / self.image_height
        return cx, cy, cw, ch

    def dict(self):
        return {
            "label": self.classname,
            "confidence": self.confidence,
            "ltx": self.x1,
            "lty": self.y1,
            "width": self.width(),
            "height": self.height(),
            "otherInfo": self.parames_vector,
            "ImageWidth": self.image_width,
            "ImageHeight": self.image_height,
        }

    def __repr__(self):
        return str({
            "label": self.classname,
            "confidence": self.confidence,
            "ltx": self.x1,
            "lty": self.y1,
            "width": self.width(),
            "height": self.height(),
            "otherInfo": self.parames_vector,
            "ImageWidth": self.image_width,
            "ImageHeight": self.image_height,
        }
        )

    def __str__(self):
        return str({
            "label": self.classname,
            "confidence": self.confidence,
            "ltx": self.x1,
            "lty": self.y1,
            "width": self.width(),
            "height": self.height(),
            "otherInfo": self.parames_vector,
            "ImageWidth": self.image_width,
            "ImageHeight": self.image_height,

        }
        )

    def __call__(self, *args, **kwargs):
        return {
            "label": self.classname,
            "confidence": self.confidence,
            "ltx": self.x1,
            "lty": self.y1,
            "width": self.width(),
            "height": self.height(),
            "otherInfo": self.parames_vector,
            "ImageWidth": self.image_width,
            "ImageHeight": self.image_height,

        }
