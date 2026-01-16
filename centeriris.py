# centeriris.py
from __future__ import annotations

import os
import math
import numpy as np
import cv2

from typing import Optional

PI = math.pi


def cascade_path(local_name: str) -> str:
    local_dir = os.path.join(os.path.dirname(__file__), "haarcascades")
    p_local = os.path.join(local_dir, local_name)
    if os.path.isfile(p_local):
        return p_local
    return os.path.join(cv2.data.haarcascades, local_name)


class CenterIris:
    """
    Python clone aligned to old CenterIris.cpp logic:
      - resize to 1024x768
      - detect biggest face
      - compute RIGHT eye bbox with fixed ratios
      - irisDetect() with:
          complement = 255 - gray
          GaussianBlur(5x5)
          threshold(..., THRESH_BINARY_INV)
          mask2 = local minima per 5x5 block
          mask cleanup with rmin gate
          partiald(..., sigma=999, n=600, part="iris")
      - output: [ix, iy, fx, fy, fw, fh] in 1024x768 coords
    """

    def __init__(self, n_poly: int = 600, max_candidates: Optional[int] = 600):
        face_path = cascade_path("haarcascade_frontalface_alt2.xml")
        self.face_cascade = cv2.CascadeClassifier(face_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Cannot load face cascade: {face_path}")

        self.n = int(n_poly)

        # Precompute trig like polygon sampling (n points)
        angles = (2.0 * PI) * (np.arange(self.n, dtype=np.float32) / self.n)
        self._sin = np.sin(angles).astype(np.float32)
        self._cos = np.cos(angles).astype(np.float32)

        # Indices used in C++ for "iris" (3 arcs)
        n = self.n
        a1 = int(math.floor(n / 8.0 + 0.5))
        a2s = int(math.floor((3 * n / 8.0 + 1) + 0.5))
        a2e = int(math.floor(5 * n / 8.0 + 0.5))
        a3s = int(math.floor((7 * n / 8.0 + 1) + 0.5))
        self._arc1 = np.arange(0, a1, dtype=np.int32)
        self._arc2 = np.arange(a2s, a2e, dtype=np.int32)
        self._arc3 = np.arange(a3s, n, dtype=np.int32)

        # sigma=999 => box filter 7 taps (exact values in C++)
        self._k_box7 = (np.ones(7, dtype=np.float32) * (1.0 / 7.0)).reshape(1, 7)

        # C++ bbox ratios (RIGHT EYE)
        self.left_r = 0.2
        self.right_r = 0.58
        self.oben_r = 0.28
        self.unten_r = 0.5

        # If not None: cap number of candidate centers for speed
        self.max_candidates = max_candidates

    def _get_bbox_features(self, img_gray_1024):
        faces = self.face_cascade.detectMultiScale(
            img_gray_1024,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
        )
        if len(faces) == 0:
            return None, None

        # biggest face
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        fx, fy, fw, fh = [int(v) for v in faces[0]]

        ex = int(fx + fw * self.left_r)
        ey = int(fy + fh * self.oben_r)
        ew = int(fw * (1.0 - self.right_r - self.left_r))
        eh = int(fh * (1.0 - self.oben_r - self.unten_r))

        face = (fx, fy, fw, fh)
        right_eye = (ex, ey, ew, eh)
        return face, right_eye

    def _lineint_fast(self, I01: np.ndarray, row: float, col: float, r: float, part: str) -> float:
        """
        C++ lineint(I, x, y, r, n, part)
          - x is ROW index, y is COL index in their code
          - vectX = x - r*sin(theta)  -> row samples
          - vectY = y + r*cos(theta)  -> col samples
          - uses nearest sampling floor(v + 0.5)
          - iris uses 3 arcs, then L = (2*s)/n
        """
        H, W = I01.shape[:2]

        rr = row - r * self._sin
        cc = col + r * self._cos

        # Out-of-image => return 0 like C++
        if (rr.max() >= (H - 1)) or (cc.max() >= (W - 1)) or (rr.min() <= 0) or (cc.min() <= 0):
            return 0.0

        rri = np.floor(rr + 0.5).astype(np.int32)
        cci = np.floor(cc + 0.5).astype(np.int32)

        if part == "pupil":
            return float(np.mean(I01[rri, cci]))

        s = 0.0
        s += float(np.sum(I01[rri[self._arc1], cci[self._arc1]]))
        s += float(np.sum(I01[rri[self._arc2], cci[self._arc2]]))
        s += float(np.sum(I01[rri[self._arc3], cci[self._arc3]]))
        # L = (2*s)/n
        return (2.0 * s) / float(self.n)

    def _partiald(self, I01: np.ndarray, row: int, col: int, rmin: float, rmax: float, part="iris"):
        """
        C++ partiald(I, x, y, rmin, rmax, sigma, n, part)
        We implement sigma=999 case (box7 smoothing).
        Returns: (max_blur, radius_at_max)
        """
        radii = np.arange(int(rmin), int(rmax), 1, dtype=np.float32)
        if radii.size < 2:
            return 0.0, float(rmin)

        Ls = []
        diffs = [0.0]  # first diff assumed 0

        prev_L = None
        for r in radii:
            L = self._lineint_fast(I01, float(row), float(col), float(r), part)
            if L == 0.0:
                break
            Ls.append(L)
            if prev_L is not None:
                diffs.append(L - prev_L)
            prev_L = L

        if len(diffs) < 3:
            return 0.0, float(rmin)

        diffs = np.array(diffs, dtype=np.float32)[None, :]  # 1 x N

        # sigma=999 => box filter length 7, anchor Point(0) like C++
        blur = cv2.filter2D(diffs, ddepth=-1, kernel=self._k_box7, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        blur = np.abs(blur).reshape(-1)

        idx = int(np.argmax(blur))
        idx = max(0, min(idx, len(radii) - 1))
        return float(blur[idx]), float(radii[idx])

    def iris_detect(self, img_eye_bgr: np.ndarray, rmin: float, rmax: float):
        """
        C++ irisDetect(imgEye, rmin, rmax) but with optional candidate cap.
        Returns (col, row) in EYE ROI coords, like C++ pushes (x,y) as (maxLoc.x, maxLoc.y)
        """
        if img_eye_bgr is None or img_eye_bgr.size == 0:
            return None

        imgEyeGray = cv2.cvtColor(img_eye_bgr, cv2.COLOR_BGR2GRAY)

        # complement = 255 - gray
        complement = (255 - imgEyeGray).astype(np.uint8)

        # GaussianBlur(5x5)
        complement = cv2.GaussianBlur(complement, (5, 5), 0)

        # threshold(..., THRESH_BINARY_INV) with thresh=0 (as in C++)
        _, binary = cv2.threshold(complement, 0, 255, cv2.THRESH_BINARY_INV)

        mask = binary.copy()
        mask2 = mask.copy()

        # imgEyeConvert: (255 - complement) / 255.5
        imgEyeConvert = (255 - complement).astype(np.float32) / 255.5

        H, W = mask.shape[:2]

        # --- mask2: for each 5x5 block, keep min location (C++ uses swapped minLoc x/y in indexing) ---
        for i in range(2, H - 2, 5):
            for j in range(2, W - 2, 5):
                small = imgEyeConvert[i - 2:i + 3, j - 2:j + 3]
                k = int(np.argmin(small))
                rr = k // 5
                cc = k % 5

                mask2[i - 2:i + 3, j - 2:j + 3] = 255

                # replicate the C++ swapped write:
                # mask2.at<uchar>((i-2)+minLoc.x, (j-2)+minLoc.y) = 0
                # where minLoc.x == cc and minLoc.y == rr
                r_write = (i - 2) + cc
                c_write = (j - 2) + rr
                if 0 <= r_write < H and 0 <= c_write < W:
                    mask2[r_write, c_write] = 0

        # --- cleanup mask with rmin gate (as C++) ---
        rmin_i = float(rmin)
        for i in range(H):
            for j in range(W):
                if (i >= (rmin_i - 2)) and (j >= (rmin_i - 2)) and (i < (H - rmin_i - 2)) and (j < (W - rmin_i - 2)):
                    if mask[i, j] != 255:
                        if mask2[i, j] == 255:
                            mask[i, j] = 255
                else:
                    mask[i, j] = 255

        # candidate centers (C++ scans i=9..H-10, j=9..W-10)
        sub = mask[9:H - 10, 9:W - 10]
        ys, xs = np.where(sub != 255)
        if ys.size == 0:
            return None

        # convert back to full coords
        ys = ys + 9
        xs = xs + 9

        # optional cap for speed (keeps behavior stable in realtime)
        if self.max_candidates is not None and ys.size > self.max_candidates:
            # choose best minima by intensity (lower intensity in imgEyeConvert = more likely)
            vals = imgEyeConvert[ys, xs]
            idx = np.argpartition(vals, self.max_candidates)[:self.max_candidates]
            ys, xs = ys[idx], xs[idx]

        best_val = -1.0
        best_rc = None  # (row, col)
        for (r, c) in zip(ys.tolist(), xs.tolist()):
            val, _ = self._partiald(imgEyeConvert, int(r), int(c), rmin, rmax, part="iris")
            if val > best_val:
                best_val = val
                best_rc = (int(r), int(c))

        if best_rc is None:
            return None

        # C++ returns [maxLoc.x, maxLoc.y] where x=col, y=row
        return (best_rc[1], best_rc[0])

    def center_iris(self, frame_bgr_640: np.ndarray):
        """
        Equivalent of CenterIris::centerIris(Mat img)
        Returns:
          ci: [ix, iy, fx, fy, fw, fh] (all in 1024x768 coords)
          resized_1024: resized frame
        """
        resized = cv2.resize(frame_bgr_640, (1024, 768), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        face, right_eye = self._get_bbox_features(gray)
        if face is None or right_eye is None:
            return None, resized

        fx, fy, fw, fh = face
        ex, ey, ew, eh = right_eye

        if ew <= 0 or eh <= 0:
            return None, resized

        eye_roi = resized[ey:ey + eh, ex:ex + ew]
        rmin = float(ew) * 0.1
        rmax = float(ew) * 0.3

        iris_xy = self.iris_detect(eye_roi, rmin, rmax)
        if iris_xy is None:
            return None, resized

        ix = int(ex + iris_xy[0])
        iy = int(ey + iris_xy[1])

        ci = [ix, iy, fx, fy, fw, fh, ex, ey, ew, eh]
        return ci, resized
