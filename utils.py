from typing import List

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from bodypix.bodypix import load_model
from skin import find_skin


def find_contours(image: np.ndarray) -> List[np.ndarray]:
    """画像配列から輪郭抽出を行う

    Parameters
    ----------
    image : numpy.ndarray
        輪郭を抽出するための2値画像配列

    Returns
    -------
    list of numpy.ndarray
        各輪郭を表す座標(ndarray)のリスト。各座標は(x, y)の2次元タプル。
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [np.squeeze(cont) for cont in contours]


def get_random_points_in_polygon(polygon, num_points):
    """polygonが表す多角形の内部からランダムにnum_points個の座標を抽出する関数

    引数:
    1. polygon - 多角形を表す配列
    2. num_points - 出力する座標の数

    出力:
    ランダムに抽出されたnum_points個の座標のリスト
    """

    poly = Polygon(polygon)
    minx, miny, maxx, maxy = poly.bounds
    points = []

    while len(points) < num_points:
        random_point = Point([np.random.randint(minx, maxx), np.random.randint(miny, maxy)])
        if poly.contains(random_point):
            points.append([random_point.x, random_point.y])
    return points


class PolkaDotMaker(object):
    def __init__(self, seed: int = 0) -> None:
        np.random.seed(seed)

    def initialize(self, mask: np.ndarray) -> None:
        self.H, self.W = mask.shape
        self.mask = np.ones_like(mask)
        self.circle = Circles()
        self.contours = Contours()

    def run(self, face_mask, swimsuit_mask, skin_mask, min_r=None):
        self.initialize(face_mask)
        # 顔のくり抜き
        self.circle.append_from_mask(face_mask)
        # 水着領域の境界を取得
        self.contours.append_from_mask(swimsuit_mask)
        # 肌領域へ円を充填
        self.filling_circle_skin(skin_mask)
        # maskへの反映
        self.draw_mask()
        num_circle = len(self.circle.centers)
        # 円の充填
        self.filling_circle(min_r)
        self.draw_mask(num_circle)
        return self.mask

    def draw_mask(self, from_=0):
        for i in range(from_, len(self.circle.centers)):
            c = self.circle.centers[i]
            r = self.circle.radiuses[i]
            cv2.circle(self.mask, c, r, 0, -1)

    def get_random_coordinates_from_mask(self, mask: np.ndarray, size):
        rows, cols = np.where(mask==1) # 1が入っている箇所の行列番号を取得
        indices = np.random.choice(len(rows), size=size, replace=False) #ランダムに50個インデックスを選択
        random_rows = rows[indices] # インデックスに対応する行を取得
        random_cols = cols[indices] # インデックスに対応する列を取得
        return np.array(list(zip(random_cols, random_rows)), dtype='uint16') # 行列番号をタプルにまとめて返す

    def filling_circle_skin(self, skin_mask, top=5):
        # 肌色領域のポリゴンを取得
        contours = Contours()
        contours.append_from_mask(skin_mask)
        # 面積上位5エリアのみを対象とする
        if len(contours.contours) > top:
            areas = np.array([cv2.contourArea(contour) for contour in contours.contours])
            skip_idx = areas.argsort()[:-top]
        else:
            skip_idx = []
        # 各エリアへの処理
        for idx, contour in enumerate(contours.contours):
            if idx in skip_idx:
                continue
            # エリアのピクセルを列挙し、ランダムに20個選択
            centers = get_random_points_in_polygon(contour, 20)
            # 半径を計算して最大になるものを採用
            rs = [self._get_radius(pt, 5) for pt in centers]
            rs = [v if v is not None else 0 for v in rs]
            index = np.argmax(rs)
            if rs[index] < 5:
                continue
            self.circle.append(centers[index], rs[index])

    def filling_circle(self, min_r=None):
        if min_r is None:
            min_r = self.circle.radiuses[0] * 0.5
        coordinates = self.get_random_coordinates_from_mask(self.mask, size=500)
        for pt in coordinates:
            r = self._get_radius(pt, min_r)
            if r is None:
                continue
            self.circle.append(pt, r)

    def _get_radius(self, pt, min_r):
        circle_out = self.circle.is_outside(pt)
        if circle_out is None:
            return
        contour_out = self.contours.is_outside(pt)
        if contour_out is None:
            return
        r = min(circle_out, contour_out)
        if r < min_r:
            return
        return r

    @staticmethod
    def chroma_key(image: np.ndarray, mask: np.ndarray, color: tuple):
        assert image.shape[:2] == mask.shape
        assert image.shape[-1] == len(color) == 3
        image[mask==1] = color
        return image


class MaskMaker(object):
    def __init__(self, bodypix_model_path) -> None:
        self.bodypix_model = load_model(bodypix_model_path)
        self._kernel = np.ones((7, 7))

    def _opening(self, image):
        return cv2.morphologyEx(
            image.astype('uint8'), cv2.MORPH_OPEN, self._kernel)

    def run(self, image, threshold=0.85):
        skin_mask = find_skin(image)/255>0.5
        skin_mask = self._opening(skin_mask)
        result = self.bodypix_model.predict_single(image)
        body_mask = result.get_mask(threshold=threshold)
        face_mask = result.get_part_mask(
            body_mask, part_names=['left_face', 'right_face'])[..., 0]
        body_mask = body_mask[..., 0]
        swimsuit_mask = np.bitwise_and(body_mask, skin_mask==0)
        swimsuit_mask = (swimsuit_mask - face_mask) > 0
        swimsuit_mask = self._opening(swimsuit_mask)
        skin_mask = (skin_mask - face_mask) > 0
        return face_mask, swimsuit_mask, skin_mask


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
        contours = [c for c in contours if len(c.shape) == 2]
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
        return self._contours_list
