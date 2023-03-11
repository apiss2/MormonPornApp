from collections import namedtuple

import cv2
import numpy as np

from bodypix.bodypix import load_model
from skin import find_skin

ResultMasks = namedtuple('masks', 'body_mask face_mask skin_mask swimsuit_mask')


def find_contours(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [np.squeeze(cont) for cont in contours]


class MaskMaker(object):
    def __init__(self, bodypix_model_path) -> None:
        self.bodypix_model = load_model(bodypix_model_path)
        self._kernel = np.ones((7, 7))

    def _opening(self, image):
        return cv2.morphologyEx(
            image.astype('uint8'), cv2.MORPH_OPEN, self._kernel)

    def run(self, image):
        skin_mask = find_skin(image)/255>0.5
        skin_mask = self._opening(skin_mask)
        result = self.bodypix_model.predict_single(image)
        body_mask = result.get_mask(threshold=0.85)
        face_mask = result.get_part_mask(
            body_mask, part_names=['left_face', 'right_face'])[..., 0]
        body_mask = body_mask[..., 0]
        swimsuit_mask = np.bitwise_and(body_mask, skin_mask==0)
        swimsuit_mask = (swimsuit_mask - face_mask) > 0
        swimsuit_mask = self._opening(swimsuit_mask)
        skin_mask = (skin_mask - face_mask) > 0
        return ResultMasks(body_mask, face_mask, skin_mask, swimsuit_mask)


class Circles(object):
    def __init__(self) -> None:
        self._centers = list()
        self._radiuses = list()

    def __len__(self):
        return len(self._centers)

    def append(self, center, radius) -> None:
        self._centers.append([int(center[0]), int(center[1])])
        self._radiuses.append(int(radius))

    def append_from_mask(self, mask: np.ndarray) -> None:
        mask = mask.astype('uint8')
        contours = find_contours(mask)
        for contour in contours:
            center, radius = cv2.minEnclosingCircle(contour)
            self.append(center, radius)

    def is_collision(self, center, radius):
        c = np.array([int(center[0]), int(center[1])])
        r = int(radius)
        dists = np.sqrt(np.sum(np.square(self.centers - c), axis=1))
        return any(dists < (self.radiuses + r))

    def is_outside(self, point):
        point = np.array(point)
        dists = np.sqrt(np.sum(np.square(self.centers - point), axis=1))
        is_inside = any((dists - self.radiuses) < 0)
        if is_inside:
            return None
        return (dists - self.radiuses).min()

    @property
    def centers(self):
        return np.array(self._centers)

    @property
    def radiuses(self):
        return np.array(self._radiuses)


class Contours(object):
    def __init__(self) -> None:
        self._points = None
        self._contours_list = list()
        self._contour_areas = list()

    def append(self, contours: list) -> None:
        self._contours_list.extend(contours)
        for contour in contours:
            area = cv2.contourArea(contour)
            self._contour_areas.append(area)
        if self._points is None:
            self._points = np.concatenate(contours)
            return
        contours.append(self._points)
        self._points = np.concatenate(contours)

    def append_from_mask(self, mask: np.ndarray):
        mask = mask.astype('uint8')
        self.append(find_contours(mask))

    def is_collision(self, center, radius):
        c = np.array([int(center[0]), int(center[1])])
        r = int(radius)
        dists = np.sqrt(np.sum(np.square(self._points - c), axis=1))
        return any(dists < r)

    def is_outside(self, point):
        rets = list()
        for contour in self._contours_list:
            ret = cv2.pointPolygonTest(contour, point, True)
            rets.append(ret)
            if ret > 0:
                return None
        return abs(max(rets))

    @property
    def contours(self):
        return np.array(self._contours_list)