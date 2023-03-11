from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bodypix.bodypix import load_model
from bodypix.image import load_image
from skin import find_skin
from utils import Circles, Contours, MaskMaker, ResultMasks


class PolkaDot(object):
    def __init__(self, masks: ResultMasks, seed: int = 0) -> None:
        np.random.seed(seed)
        self.masks = masks
        self.H, self.W = self.masks.swimsuit_mask.shape[:2]
        self.circle = Circles()
        self.contours = Contours()

    def run(self, min_r=None):
        # 顔のくり抜き
        self.circle.append_from_mask(self.masks.face_mask)
        # 水着領域の境界を取得
        self.contours.append_from_mask(self.masks.swimsuit_mask)
        # 肌領域へ円を充填
        self.filling_circle_skin()
        # 円の充填
        self.filling_circle(min_r)

    def filling_circle_skin(self):
        # 肌色領域
        contours = Contours()
        contours.append_from_mask(self.masks.skin_mask)
        # 面積上位5エリアのみを対象とする
        if len(contours.contours) > 5:
            areas = np.array([cv2.contourArea(contour) for contour in contours._contours_list])
            skip_idx = areas.argsort()[:-5]
        else:
            skip_idx = []
        # 各エリアへの処理
        for idx, contour in enumerate(contours.contours):
            if idx in skip_idx:
                continue
            # エリアのピクセルを列挙し、ランダムに20個選択
            empty = np.zeros_like(self.masks.skin_mask, dtype='uint8')
            mask = cv2.fillPoly(empty, pts=[contour], color=1)
            x, y = np.where(mask)
            idx = np.random.choice([i for i in range(len(x))], 50)
            centers = np.array([y[idx], x[idx]], dtype='uint16').T
            # 半径を計算して最大になるものを採用
            rs = [self._get_radius(pt, 5) for pt in centers]
            max_r, max_c = 0, 0
            for c, r in zip(centers, rs):
                if r is None:
                    continue
                if r > max_r:
                    max_r, max_c = r, c
            if max_r < 5:
                continue
            self.circle.append(max_c, max_r)

    def filling_circle(self, min_r=None):
        if min_r is None:
            min_r = self.circle.radiuses[0] * 0.5
        num = 0
        while True:
            num += 1
            pt = self._generate_random_points()
            r = self._get_radius(pt, min_r)
            if r is None:
                continue
            self.circle.append(pt, r)
            if num > 500:
                return

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

    def _generate_random_points(self):
        h = np.random.randint(0, self.H)
        w = np.random.randint(0, self.W)
        return w, h


if __name__ == '__main__':
    model_path = './tflitemodels/mobilenet-float-multiplier-050-stride16-float16.tflite'
    maker = MaskMaker(model_path)

    img_path = './images/53694183b_37_d_500.jpg'
    image_array = load_image(img_path, max_size=800)
    masks = maker.run(image_array)

    dot = PolkaDot(masks)
    dot.run(min_r=20)

    for i in range(len(dot.circle.centers)):
        c = dot.circle.centers[i]
        r = dot.circle.radiuses[i]
        cv2.circle(image_array, c, r, (255, 0, 0), 2)
    plt.imshow(image_array)
    plt.show()
