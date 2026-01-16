import os
import math
import time
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QGridLayout, QMessageBox, QFileDialog,
    QSpinBox
)

# ============================================================
# 1) CONFIG
# ============================================================
class Config:
    CAM_W, CAM_H = 640, 480
    W_RESIZE, H_RESIZE = 1024, 768

    # Intrinsics (idéalement: charge ton .yml)
    FX, FY = 600.0, 600.0
    CX, CY = 320.0, 240.0

    # Blur scale (sigma0) utilisé pour "ajouter" un blur connu
    SIGMA0 = 2.0

    # Paramètres blur->depth (Eq. 9 du papier)
    F_MM = 4.033      # F (focal length) en mm
    f_MM = 4.0676     # f (distance lentille->image plane) en mm
    F_NUM = 2.8       # f-number
    K_CAL = 0.0013    # constante k
    U_MM = 500.0      # u (position de focus parfaite) - utilisé dans l'article (expériences)

    NB_CLASSES = 9

    @staticmethod
    def cascade_path(local_name: str) -> str:
        local_dir = os.path.join(os.path.dirname(__file__), "haarcascades")
        p_local = os.path.join(local_dir, local_name)
        if os.path.isfile(p_local):
            return p_local
        return os.path.join(cv2.data.haarcascades, local_name)

    @staticmethod
    def load_intrinsics_from_yml(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return False
        cm = fs.getNode("camera_matrix").mat()
        fs.release()
        if cm is None or cm.shape != (3, 3):
            return False
        Config.FX = float(cm[0, 0])
        Config.FY = float(cm[1, 1])
        Config.CX = float(cm[0, 2])
        Config.CY = float(cm[1, 2])
        return True


# ============================================================
# 2) OUTILS I/O
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_csv_line(path: str, arr):
    s = ",".join([f"{float(x):.10g}" for x in arr])
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def read_features_file(path: str, expected_dim=8):
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    if not txt:
        return None
    toks = [t for t in txt.replace("\n", ",").split(",") if t.strip() != ""]
    vals = np.array([float(t) for t in toks], dtype=np.float64)
    if vals.size % expected_dim == 0:
        return vals.reshape(-1, expected_dim)
    if vals.size >= expected_dim:
        return vals[:expected_dim][None, :]
    return None


# ============================================================
# 3) PREDICT (one-vs-one)
# ============================================================
def _logistic_safe(s: float) -> float:
    if s >= 0.0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)

def _load_vector_csv(path: str):
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    if not txt:
        return None
    toks = [t for t in txt.replace("\n", ",").split(",") if t.strip() != ""]
    return np.array([float(t) for t in toks], dtype=np.float64)

class VBMLRPredictOneVsOne:
    def __init__(self, base_dir: str, nb_classes: int = 9):
        self.base_dir = base_dir
        self.C = nb_classes
        self.disc = [[None for _ in range(self.C)] for _ in range(self.C)]
        self.reload()

    def reload(self):
        for j in range(1, self.C + 1):
            for i in range(1, self.C + 1):
                if i == j:
                    continue
                path = os.path.join(self.base_dir, f"disc_{j}_{i}.txt")
                self.disc[j-1][i-1] = _load_vector_csv(path)

    @staticmethod
    def _dot_bias_last(w, x):
        d = min(len(x), len(w) - 1)
        return float(np.dot(w[:d], x[:d]) + w[d])

    def predict(self, feats):
        votes = np.zeros(self.C, dtype=np.float64)
        for j in range(1, self.C + 1):
            vj = 0.0
            for i in range(1, self.C + 1):
                if i == j:
                    continue
                w = self.disc[j-1][i-1]
                if w is None:
                    continue
                sm = self._dot_bias_last(w, feats)
                p = _logistic_safe(sm)
                vj += 1.0 if (p <= 0.5) else 0.0
            votes[j-1] = vj
        return int(np.argmax(votes)) + 1


# ============================================================
# 4) Face / Eyes / Nose + Iris (paper-aligned spirit)
# ============================================================
class FaceEyeNoseDetector:
    """
    - Face (Viola-Jones)
    - Eyes (Viola-Jones)
    - Nose (cascade si dispo; sinon fallback géométrique)
    - Iris centre via radial derivative search (ton approche)
    + Stabilisation temporelle (iris + nez)
    """
    def __init__(self):
        self.face = cv2.CascadeClassifier(Config.cascade_path("haarcascade_frontalface_alt2.xml"))
        self.eye  = cv2.CascadeClassifier(Config.cascade_path("haarcascade_eye_tree_eyeglasses.xml"))

        nose_path = Config.cascade_path("haarcascade_mcs_nose.xml")
        self.nose = cv2.CascadeClassifier(nose_path)
        if self.face.empty() or self.eye.empty():
            raise RuntimeError("Cannot load cascades (face/eye). Check haarcascade files.")
        if self.nose.empty():
            self.nose = None

        # iris detector precompute
        self._n_poly = 600
        ang = (2.0 * math.pi) * (np.arange(self._n_poly, dtype=np.float32) / self._n_poly)
        self._sin = np.sin(ang).astype(np.float32)
        self._cos = np.cos(ang).astype(np.float32)

        n = self._n_poly
        a1 = int(math.floor(n/8.0 + 0.5))
        a2s = int(math.floor((3*n/8.0 + 1) + 0.5))
        a2e = int(math.floor(5*n/8.0 + 0.5))
        a3s = int(math.floor((7*n/8.0 + 1) + 0.5))
        idx1 = np.arange(0, a1, dtype=np.int32)
        idx2 = np.arange(a2s, a2e, dtype=np.int32)
        idx3 = np.arange(a3s, n, dtype=np.int32)
        self._iris_arc_idx = np.concatenate([idx1, idx2, idx3])
        self._k_box7 = (np.ones(7, dtype=np.float32) / 7.0)

        # temporal stabilization
        self.prev_iris_L = None
        self.prev_iris_R = None
        self.prev_nose_c = None

        self.ema_alpha = 0.65
        self.max_jump_px_iris = 25
        self.max_jump_px_nose = 35

    def _stabilize_point(self, pt, prev, max_jump):
        if pt is None:
            return prev
        if prev is None:
            return pt
        dx = pt[0] - prev[0]
        dy = pt[1] - prev[1]
        if (dx*dx + dy*dy) > (max_jump * max_jump):
            return prev
        ax = self.ema_alpha * prev[0] + (1.0 - self.ema_alpha) * pt[0]
        ay = self.ema_alpha * prev[1] + (1.0 - self.ema_alpha) * pt[1]
        return (int(ax), int(ay))

    def detect_face(self, gray_1024):
        faces = self.face.detectMultiScale(
            gray_1024, scaleFactor=1.1, minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        return tuple(map(int, faces[0]))

    def detect_eyes_in_face(self, gray_1024, face_bbox):
        fx, fy, fw, fh = face_bbox

        # search upper half
        y_top = fy + int(0.12 * fh)
        y_bot = fy + int(0.60 * fh)
        x_left = fx + int(0.10 * fw)
        x_right = fx + int(0.90 * fw)

        roi = gray_1024[y_top:y_bot, x_left:x_right]
        if roi.size == 0:
            return None, None

        roi = cv2.equalizeHist(roi)

        eyes = self.eye.detectMultiScale(
            roi, scaleFactor=1.08, minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(30, 30)
        )
        if len(eyes) == 0:
            return None, None

        eyes_abs = []
        for (ex, ey, ew, eh) in eyes:
            eyes_abs.append((x_left + ex, y_top + ey, ew, eh))

        # keep a few best
        eyes_abs = sorted(eyes_abs, key=lambda r: r[2]*r[3], reverse=True)[:6]

        # split by face midline if possible
        face_mid_x = fx + fw * 0.5
        left_cand  = [e for e in eyes_abs if (e[0] + e[2]*0.5) < face_mid_x]
        right_cand = [e for e in eyes_abs if (e[0] + e[2]*0.5) >= face_mid_x]

        left_eye = None
        right_eye = None

        if len(left_cand) > 0:
            left_eye = sorted(left_cand, key=lambda r: r[2]*r[3], reverse=True)[0]
        if len(right_cand) > 0:
            right_eye = sorted(right_cand, key=lambda r: r[2]*r[3], reverse=True)[0]

        # fallback: if midline split fails, use leftmost & rightmost
        if left_eye is None and right_eye is None:
            eyes_lr = sorted(eyes_abs, key=lambda r: r[0])
            if len(eyes_lr) >= 2:
                left_eye, right_eye = eyes_lr[0], eyes_lr[-1]
            else:
                left_eye, right_eye = eyes_lr[0], None

        # ensure order
        if left_eye is not None and right_eye is not None and left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye

        return left_eye, right_eye

    def detect_nose_in_face(self, gray_1024, face_bbox):
        fx, fy, fw, fh = face_bbox

        # if cascade unavailable -> fallback point
        if self.nose is None:
            # approx nose around center-lower
            nx = int(fx + 0.50 * fw)
            ny = int(fy + 0.58 * fh)
            return (nx, ny, 0, 0)  # we will use center only

        roi = gray_1024[fy:fy+fh, fx:fx+fh]  # square-ish ROI helps
        if roi.size == 0:
            return None
        roi_eq = cv2.equalizeHist(roi)

        noses = self.nose.detectMultiScale(
            roi_eq, scaleFactor=1.2, minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(25, 25)
        )
        if len(noses) == 0:
            return None

        noses = sorted(noses, key=lambda r: r[2]*r[3], reverse=True)
        nx, ny, nw, nh = noses[0]

        # map back to full image
        return (fx+nx, fy+ny, nw, nh)

    def _partiald_fast(self, I_float01, cx, cy, rmin, rmax, part="iris"):
        H, W = I_float01.shape[:2]
        radii = np.arange(int(rmin), int(rmax), 1, dtype=np.float32)
        if radii.size < 2:
            return 0.0, float(rmin)
        rr = radii[:, None]
        map_x = (cx + rr * self._cos[None, :]).astype(np.float32)
        map_y = (cy + rr * self._sin[None, :]).astype(np.float32)
        if (map_x.min() < 1) or (map_y.min() < 1) or (map_x.max() > (W-2)) or (map_y.max() > (H-2)):
            return 0.0, float(rmin)

        samples = cv2.remap(I_float01, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if part == "pupil":
            L = samples.mean(axis=1)
        else:
            L = samples[:, self._iris_arc_idx].mean(axis=1)

        diff = np.empty_like(L, dtype=np.float32)
        diff[0] = 0.0
        diff[1:] = (L[1:] - L[:-1]).astype(np.float32)
        smooth = np.convolve(diff, self._k_box7, mode="same")
        smooth = np.abs(smooth)
        idx = int(np.argmax(smooth))
        idx = max(0, min(idx, radii.size - 1))
        return float(smooth[idx]), float(radii[idx])

    def iris_detect(self, eye_bgr, rmin, rmax):
        if eye_bgr is None or eye_bgr.size == 0:
            return None

        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        complement = (255 - gray).astype(np.uint8)
        complement = cv2.GaussianBlur(complement, (5, 5), 0)
        _, binary = cv2.threshold(complement, 0, 255, cv2.THRESH_BINARY_INV)

        mask = binary.copy()
        mask2 = mask.copy()
        imgEyeConvert = (255 - complement).astype(np.float32) / 255.5
        H, W = mask.shape[:2]

        for i in range(2, H-2, 5):
            for j in range(2, W-2, 5):
                patch = imgEyeConvert[i-2:i+3, j-2:j+3]
                minLoc = np.unravel_index(int(np.argmin(patch)), patch.shape)
                mask2[i-2:i+3, j-2:j+3] = 255
                mask2[(i-2)+minLoc[0], (j-2)+minLoc[1]] = 0

        for i in range(H):
            for j in range(W):
                if (i >= (rmin-2)) and (j >= (rmin-2)) and (i < (H - rmin - 2)) and (j < (W - rmin - 2)):
                    if mask[i, j] != 255:
                        if mask2[i, j] == 255:
                            mask[i, j] = 255
                else:
                    mask[i, j] = 255

        ys, xs = np.where(mask != 255)
        if ys.size == 0:
            return None

        topK = 220
        vals = imgEyeConvert[ys, xs]
        if ys.size > topK:
            idx = np.argpartition(vals, topK)[:topK]
            ys, xs = ys[idx], xs[idx]

        best_val = -1.0
        best_xy = None
        for (y, x) in zip(ys, xs):
            val, _ = self._partiald_fast(imgEyeConvert, float(x), float(y), rmin, rmax, part="iris")
            if val > best_val:
                best_val = val
                best_xy = (int(x), int(y))
        return best_xy

    def detect_all(self, frame_bgr_640):
        resized = cv2.resize(frame_bgr_640, (Config.W_RESIZE, Config.H_RESIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        face_bbox = self.detect_face(gray)
        if face_bbox is None:
            return None

        left_eye, right_eye = self.detect_eyes_in_face(gray, face_bbox)
        nose_bbox = self.detect_nose_in_face(gray, face_bbox)

        # canthi (inner corners)
        canthus_L = None
        canthus_R = None
        if left_eye is not None:
            lx, ly, lw, lh = left_eye
            canthus_L = (lx + lw, ly + lh // 2)  # inner corner for left eye
        if right_eye is not None:
            rx, ry, rw, rh = right_eye
            canthus_R = (rx, ry + rh // 2)       # inner corner for right eye

        # iris detection with better ROI (remove top region -> reduces eyebrow noise)
        iris_L = None
        iris_R = None

        if left_eye is not None:
            lx, ly, lw, lh = left_eye
            y1 = ly + int(0.20 * lh)
            y2 = ly + int(0.95 * lh)
            x1 = lx + int(0.05 * lw)
            x2 = lx + int(0.95 * lw)
            eye_roi = resized[y1:y2, x1:x2]
            rmin = int(min(lw, lh) * 0.10)
            rmax = int(min(lw, lh) * 0.35)
            iris_xy = self.iris_detect(eye_roi, rmin, rmax)
            if iris_xy is not None:
                iris_L = (x1 + iris_xy[0], y1 + iris_xy[1])

        if right_eye is not None:
            rx, ry, rw, rh = right_eye
            y1 = ry + int(0.20 * rh)
            y2 = ry + int(0.95 * rh)
            x1 = rx + int(0.05 * rw)
            x2 = rx + int(0.95 * rw)
            eye_roi = resized[y1:y2, x1:x2]
            rmin = int(min(rw, rh) * 0.10)
            rmax = int(min(rw, rh) * 0.35)
            iris_xy = self.iris_detect(eye_roi, rmin, rmax)
            if iris_xy is not None:
                iris_R = (x1 + iris_xy[0], y1 + iris_xy[1])

        # stabilize iris points
        iris_L = self._stabilize_point(iris_L, self.prev_iris_L, self.max_jump_px_iris)
        iris_R = self._stabilize_point(iris_R, self.prev_iris_R, self.max_jump_px_iris)
        self.prev_iris_L = iris_L
        self.prev_iris_R = iris_R

        # nose center (stabilized); if bbox missing, fallback from geometry
        nx_c = None
        if nose_bbox is not None:
            nx, ny, nw, nh = nose_bbox
            if nw > 0 and nh > 0:
                nx_c = (nx + nw // 2, ny + nh // 2)
            else:
                nx_c = (nx, ny)

        if nx_c is None:
            fx, fy, fw, fh = face_bbox
            # if canthi exist, use them
            if canthus_L is not None and canthus_R is not None:
                midx = int(0.5 * (canthus_L[0] + canthus_R[0]))
                midy = int(0.5 * (canthus_L[1] + canthus_R[1]) + 0.28 * fh)
                nx_c = (midx, midy)
            else:
                nx_c = (int(fx + 0.50 * fw), int(fy + 0.58 * fh))

        nx_c = self._stabilize_point(nx_c, self.prev_nose_c, self.max_jump_px_nose)
        self.prev_nose_c = nx_c

        return {
            "resized": resized,
            "gray": gray,
            "face": face_bbox,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose_bbox,
            "nose_c": nx_c,
            "iris_L": iris_L,
            "iris_R": iris_R,
            "canthus_L": canthus_L,
            "canthus_R": canthus_R,
        }


# ============================================================
# 5) Blur -> sigma map -> depth -> Head pose + iris displacement
# ============================================================
class BlurFeatureExtractor:
    def __init__(self):
        self.det = FaceEyeNoseDetector()

    @staticmethod
    def blur_sigma_map(frame_bgr_640):
        """
        Paper-aligned (Eq. 15):
          ratio = ||∇I(σ1)|| / ||∇I(σ)|| = σ^2 / (σ^2 + σ0^2)
        where:
          I(σ)  = acquired image (unknown blur σ)
          I(σ1) = acquired image convolved with Gaussian(σ0)
        => σ = σ0 * sqrt(1/ratio - 1)

        Implementation:
          G0 = ||∇I|| on original
          G1 = ||∇(GaussianBlur(I, σ0))||
          ratio = G1 / G0
        """
        gray = cv2.cvtColor(frame_bgr_640, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sigma0 = float(Config.SIGMA0)

        blur = cv2.GaussianBlur(gray, (0, 0), sigma0)

        gX0 = cv2.Sobel(gray,  cv2.CV_32F, 1, 0, ksize=3)
        gY0 = cv2.Sobel(gray,  cv2.CV_32F, 0, 1, ksize=3)
        gX1 = cv2.Sobel(blur,  cv2.CV_32F, 1, 0, ksize=3)
        gY1 = cv2.Sobel(blur,  cv2.CV_32F, 0, 1, ksize=3)

        G0 = cv2.magnitude(gX0, gY0) + 1e-8
        G1 = cv2.magnitude(gX1, gY1) + 1e-8

        ratio = G1 / G0  # in (0,1] ideally
        ratio = np.clip(ratio, 1e-4, 1.0)

        sig = sigma0 * np.sqrt(np.maximum((1.0 / ratio) - 1.0, 0.0)).astype(np.float32)

        # smooth a bit
        sig = cv2.medianBlur(sig, 3)
        return sig

    @staticmethod
    def depth_from_sigma_mm(sigma):
        """
        Eq. (9) du papier (blur->depth) en mm:
          z = (F f) / (f - F ± k F N σ)
        Dans le papier, le signe dépend de z>=u ou pas.
        Ici, on garde une version stable:
          denom = (f - F) - k F N σ
          z = abs((F f) / denom)
        """
        F = float(Config.F_MM)
        f = float(Config.f_MM)
        N = float(Config.F_NUM)
        k = float(Config.K_CAL)

        denom = (f - F) - (k * F * N * float(sigma))
        if abs(denom) < 1e-12:
            return None
        z = abs((F * f) / denom)
        return z

    @staticmethod
    def uv_to_xyz_cm(u, v, z_mm):
        if z_mm is None:
            return None
        x_mm = (z_mm / float(Config.FX)) * (float(u) - float(Config.CX))
        y_mm = (z_mm / float(Config.FY)) * (float(v) - float(Config.CY))
        return np.array([x_mm/10.0, y_mm/10.0, z_mm/10.0], dtype=np.float64)

    @staticmethod
    def sample_sigma(sigmas_640, u_1024, v_1024, win=9):
        if sigmas_640 is None:
            return None
        u = int(u_1024 * (Config.CAM_W / Config.W_RESIZE))
        v = int(v_1024 * (Config.CAM_H / Config.H_RESIZE))

        H, W = sigmas_640.shape[:2]
        r = win // 2
        x1 = max(0, u - r)
        x2 = min(W, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(H, v + r + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = sigmas_640[y1:y2, x1:x2]
        return float(np.mean(patch))

    @staticmethod
    def normalize_plus_one(feats8):
        feats8 = np.asarray(feats8, dtype=np.float64)
        n = float(np.linalg.norm(feats8))
        if n < 1e-12:
            return feats8 * 0.0
        return (feats8 / n) + 1.0

    def extract_features(self, frame_bgr_640):
        det = self.det.detect_all(frame_bgr_640)
        if det is None:
            return None, None, {"reason": "no_face"}

        resized = det["resized"]
        face = det["face"]
        nose_c = det.get("nose_c", None)
        iris_L = det.get("iris_L", None)
        iris_R = det.get("iris_R", None)
        cL = det.get("canthus_L", None)
        cR = det.get("canthus_R", None)

        if nose_c is None or cR is None:
            return None, resized, {"reason": "missing_points", "det": det}

        # Need at least one iris
        if iris_L is None and iris_R is None:
            return None, resized, {"reason": "no_iris", "det": det}

        # sigma map on 640x480
        sigmas = self.blur_sigma_map(frame_bgr_640)

        # A,B,C in IMAGE (1024)
        A_uv = (int(nose_c[0]), int(nose_c[1]))  # nose tip approx

        # If left canthus missing, fallback in face
        if cL is not None:
            B_uv = cL
        else:
            fx, fy, fw, fh = face
            B_uv = (fx + int(0.35*fw), fy + int(0.45*fh))

        C_uv = cR

        head_uv = ((B_uv[0] + C_uv[0]) / 2.0, (B_uv[1] + C_uv[1]) / 2.0)

        sA = self.sample_sigma(sigmas, A_uv[0], A_uv[1], win=11)
        sB = self.sample_sigma(sigmas, B_uv[0], B_uv[1], win=11)
        sC = self.sample_sigma(sigmas, C_uv[0], C_uv[1], win=11)
        if sA is None or sB is None or sC is None:
            return None, resized, {"reason": "sigma_sample_fail", "det": det}

        zA = self.depth_from_sigma_mm(sA)
        zB = self.depth_from_sigma_mm(sB)
        zC = self.depth_from_sigma_mm(sC)
        if zA is None or zB is None or zC is None:
            return None, resized, {"reason": "depth_fail", "det": det, "sig": (sA, sB, sC)}

        A = self.uv_to_xyz_cm(A_uv[0], A_uv[1], zA)
        B = self.uv_to_xyz_cm(B_uv[0], B_uv[1], zB)
        C = self.uv_to_xyz_cm(C_uv[0], C_uv[1], zC)
        if A is None or B is None or C is None:
            return None, resized, {"reason": "xyz_fail", "det": det}

        AB = (B - A)
        AC = (C - A)
        cross = np.cross(AB, AC)
        cn = float(np.linalg.norm(cross))
        if cn < 1e-12:
            R = np.zeros(3, dtype=np.float64)
        else:
            R = cross / cn

        T = (A + B + C) / 3.0

        # iris used (average if both available)
        if iris_L is not None and iris_R is not None:
            iris_u = 0.5 * (iris_L[0] + iris_R[0])
            iris_v = 0.5 * (iris_L[1] + iris_R[1])
        elif iris_R is not None:
            iris_u, iris_v = iris_R
        else:
            iris_u, iris_v = iris_L

        delta_x = float(iris_u - head_uv[0])
        delta_y = float(iris_v - head_uv[1])

        feats8_raw = np.array([T[0], T[1], T[2], R[0], R[1], R[2], delta_x, delta_y], dtype=np.float64)
        feats8 = self.normalize_plus_one(feats8_raw)

        dbg = {
            "det": det,
            "sigmas": sigmas,
            "A_uv": A_uv, "B_uv": B_uv, "C_uv": C_uv,
            "head_uv": head_uv,
            "T": T, "R": R,
            "delta": (delta_x, delta_y),
            "raw": feats8_raw,
            "feats": feats8,
            "sigma_pts": (sA, sB, sC),
            "z_mm": (zA, zB, zC),
        }
        return feats8, resized, dbg


# ============================================================
# 6) VBMLR TRAINER
# ============================================================
def _lambda_xi(xi):
    xi = np.asarray(xi, dtype=np.float64)
    out = np.empty_like(xi)
    small = xi < 1e-9
    out[small] = 1.0 / 8.0
    xs = xi[~small]
    out[~small] = np.tanh(xs / 2.0) / (4.0 * xs)
    return out

def vb_logistic_fit(X, y, alpha=1.0, max_iter=200, tol=1e-6):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    N, D = X.shape
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    Db = D + 1

    mu = np.zeros(Db, dtype=np.float64)
    Sigma = np.eye(Db, dtype=np.float64) / alpha
    xi = np.ones(N, dtype=np.float64)

    I = np.eye(Db, dtype=np.float64)

    for _ in range(max_iter):
        lam = _lambda_xi(xi)
        XL = Xb * (np.sqrt(2.0 * lam)[:, None])
        A = (alpha * I) + (XL.T @ XL)
        Sigma_new = np.linalg.inv(A)

        t = (y - 0.5)
        mu_new = Sigma_new @ (Xb.T @ t)

        S = Sigma_new + np.outer(mu_new, mu_new)
        xi_new = np.sqrt(np.einsum("nd,dd,nd->n", Xb, S, Xb) + 1e-12)

        dm = np.linalg.norm(mu_new - mu) / (np.linalg.norm(mu) + 1e-9)
        mu, Sigma, xi = mu_new, Sigma_new, xi_new
        if dm < tol:
            break

    return mu

def train_vbmlr_all_pairs(dataset_dir: str, out_dir: str, nb_classes=9, min_samples=10, status_cb=None):
    ensure_dir(out_dir)
    Xc = {}
    for c in range(1, nb_classes+1):
        p = os.path.join(dataset_dir, "features", f"class{c}.txt")
        X = read_features_file(p, expected_dim=8)
        if X is None or X.shape[0] < min_samples:
            raise RuntimeError(f"Pas assez de samples pour class{c}: {p}")
        Xc[c] = X

    total = nb_classes * (nb_classes - 1)
    k = 0

    for j in range(1, nb_classes+1):
        for i in range(1, nb_classes+1):
            if i == j:
                continue
            k += 1
            if status_cb:
                status_cb(f"Training disc_{j}_{i}.txt ({k}/{total})")

            Xj = Xc[j]
            Xi = Xc[i]
            X = np.vstack([Xj, Xi])
            y = np.hstack([np.zeros(Xj.shape[0]), np.ones(Xi.shape[0])])  # j=0, i=1

            w = vb_logistic_fit(X, y, alpha=1.0, max_iter=200, tol=1e-6)

            outp = os.path.join(out_dir, f"disc_{j}_{i}.txt")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(",".join([f"{float(v):.12g}" for v in w]) + "\n")


# ============================================================
# 7) UI
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Tracking (Paper-aligned Blur Method + VBMLR)")

        self.extractor = BlurFeatureExtractor()
        self.predictor = None

        self.dataset_root = None
        self.disc_dir = None

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)

        self.mode = "IDLE"
        self.curr_class = 1
        self.samples_in_class = 0
        self.prediction = -1
        self._t = 0
        self.cooldown = 0

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        layout.addLayout(left, 1)

        self.btn_pick_root = QPushButton("Select dataset folder")
        self.btn_pick_root.clicked.connect(self.pick_dataset_root)
        left.addWidget(self.btn_pick_root)

        self.btn_load_intr = QPushButton("Load intrinsics .yml")
        self.btn_load_intr.clicked.connect(self.load_intrinsics)
        left.addWidget(self.btn_load_intr)

        left.addSpacing(10)

        self.btn_start_cam = QPushButton("Start Webcam")
        self.btn_stop_cam = QPushButton("Stop Webcam")
        self.btn_start_cam.clicked.connect(self.start_webcam)
        self.btn_stop_cam.clicked.connect(self.stop_webcam)
        left.addWidget(self.btn_start_cam)
        left.addWidget(self.btn_stop_cam)

        left.addSpacing(10)

        self.btn_new_acq = QPushButton("New acquisition (1..9)")
        self.btn_new_acq.clicked.connect(self.new_acquisition)
        left.addWidget(self.btn_new_acq)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Samples/class"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(5, 500)
        self.spin_samples.setValue(40)
        hl.addWidget(self.spin_samples)
        left.addLayout(hl)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Cooldown(frames)"))
        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(0, 60)
        self.spin_cooldown.setValue(8)
        hl2.addWidget(self.spin_cooldown)
        left.addLayout(hl2)

        left.addSpacing(10)

        self.btn_train = QPushButton("Train VBMLR (build disc_*.txt)")
        self.btn_train.clicked.connect(self.train_vbmlr)
        left.addWidget(self.btn_train)

        self.btn_load_disc = QPushButton("Load disc vectors")
        self.btn_load_disc.clicked.connect(self.load_disc_vectors)
        left.addWidget(self.btn_load_disc)

        self.btn_live = QPushButton("Real Time Tracking")
        self.btn_live.clicked.connect(self.enable_live)
        left.addWidget(self.btn_live)

        left.addStretch(1)

        center_right = QVBoxLayout()
        layout.addLayout(center_right, 4)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(900, 500)
        self.video.setStyleSheet("background-color:#222;")
        center_right.addWidget(self.video, 6)

        bottom = QHBoxLayout()
        center_right.addLayout(bottom, 3)

        settings = QGroupBox("Status")
        s_lay = QVBoxLayout(settings)
        self.lbl_status = QLabel("MODE: IDLE")
        s_lay.addWidget(self.lbl_status)
        self.lbl_info = QLabel("—")
        s_lay.addWidget(self.lbl_info)
        bottom.addWidget(settings, 2)

        grid_box = QGroupBox("")
        grid = QGridLayout(grid_box)
        self.grid_btns = []
        k = 1
        for r in range(3):
            for c in range(3):
                btn = QPushButton(str(k))
                btn.setFixedSize(80, 80)
                btn.setEnabled(False)
                self.grid_btns.append(btn)
                grid.addWidget(btn, r, c)
                k += 1
        bottom.addWidget(grid_box, 2)

    def pick_dataset_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select / create dataset folder")
        if not d:
            return
        self.dataset_root = d
        ensure_dir(os.path.join(d, "images"))
        ensure_dir(os.path.join(d, "features"))
        for c in range(1, Config.NB_CLASSES+1):
            ensure_dir(os.path.join(d, "images", f"class{c}"))
        self.lbl_info.setText(f"dataset_root = {d}")

    def load_intrinsics(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select camera .yml", filter="YML (*.yml *.yaml);;All Files (*)")
        if not p:
            return
        ok = Config.load_intrinsics_from_yml(p)
        if ok:
            QMessageBox.information(self, "Intrinsics loaded",
                                    f"FX={Config.FX:.3f} FY={Config.FY:.3f} CX={Config.CX:.3f} CY={Config.CY:.3f}")
        else:
            QMessageBox.warning(self, "Failed", "Impossible de lire camera_matrix depuis ce .yml")

    def start_webcam(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_H)
        self.timer.start(10)
        self.lbl_info.setText("Webcam started")

    def stop_webcam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.lbl_info.setText("Webcam stopped")

    def new_acquisition(self):
        if self.dataset_root is None:
            QMessageBox.warning(self, "Dataset missing", "Sélectionne d'abord un dataset folder.")
            return
        self.mode = "RECORDING"
        self.curr_class = 1
        self.samples_in_class = 0
        self.prediction = -1
        self.cooldown = 0
        self.lbl_info.setText("Recording class 1...")

        for c in range(1, Config.NB_CLASSES+1):
            fp = os.path.join(self.dataset_root, "features", f"class{c}.txt")
            open(fp, "w", encoding="utf-8").close()

    def train_vbmlr(self):
        if self.dataset_root is None:
            QMessageBox.warning(self, "Dataset missing", "Sélectionne d'abord un dataset folder.")
            return
        out_dir = os.path.join(self.dataset_root, "disc")
        self.disc_dir = out_dir

        try:
            def cb(msg):
                self.lbl_info.setText(msg)
                QApplication.processEvents()

            train_vbmlr_all_pairs(
                dataset_dir=self.dataset_root,
                out_dir=out_dir,
                nb_classes=Config.NB_CLASSES,
                min_samples=10,
                status_cb=cb
            )
        except Exception as e:
            QMessageBox.critical(self, "Training failed", str(e))
            return

        QMessageBox.information(self, "Training done", f"disc_*.txt created in:\n{out_dir}")

    def load_disc_vectors(self):
        base = QFileDialog.getExistingDirectory(self, "Select disc folder (contains disc_*.txt)")
        if not base:
            return
        self.disc_dir = base
        self.predictor = VBMLRPredictOneVsOne(base_dir=base, nb_classes=Config.NB_CLASSES)

        loaded = 0
        for j in range(1, Config.NB_CLASSES+1):
            for i in range(1, Config.NB_CLASSES+1):
                if i == j:
                    continue
                if self.predictor.disc[j-1][i-1] is not None:
                    loaded += 1
        QMessageBox.information(self, "Disc loaded",
                                f"Loaded {loaded}/{Config.NB_CLASSES*(Config.NB_CLASSES-1)} disc vectors.")

    def enable_live(self):
        if self.predictor is None:
            QMessageBox.warning(self, "Predictor missing", "Charge d'abord un dossier disc (disc_*.txt).")
            return
        self.mode = "PREDICTING"
        self.lbl_info.setText("Live predicting...")

    def _update_grid_colors(self):
        for i, btn in enumerate(self.grid_btns, start=1):
            col = "#e6e6e6"
            if self.mode == "RECORDING":
                if i == self.curr_class:
                    col = "#ffa500"
                elif i < self.curr_class:
                    col = "#b0ffb0"
            elif self.mode == "PREDICTING" and i == self.prediction:
                col = "#00ff00"
            btn.setStyleSheet(f"background-color:{col};")

    def on_tick(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        feats, frame_1024, dbg = self.extractor.extract_features(frame)
        if frame_1024 is None:
            frame_1024 = cv2.resize(frame, (Config.W_RESIZE, Config.H_RESIZE))

        if self.mode == "RECORDING":
            if self.cooldown > 0:
                self.cooldown -= 1
            else:
                if feats is not None:
                    img_dir = os.path.join(self.dataset_root, "images", f"class{self.curr_class}")
                    ensure_dir(img_dir)
                    idx = self.samples_in_class + 1
                    img_path = os.path.join(img_dir, f"img_{idx:04d}.png")
                    cv2.imwrite(img_path, frame)

                    feat_path = os.path.join(self.dataset_root, "features", f"class{self.curr_class}.txt")
                    write_csv_line(feat_path, feats)

                    self.samples_in_class += 1
                    self.cooldown = int(self.spin_cooldown.value())

                    if self.samples_in_class >= int(self.spin_samples.value()):
                        self.curr_class += 1
                        self.samples_in_class = 0
                        self.cooldown = int(self.spin_cooldown.value())

                        if self.curr_class > Config.NB_CLASSES:
                            self.mode = "IDLE"
                            self.lbl_info.setText("Acquisition finished. Now train VBMLR.")
                        else:
                            self.lbl_info.setText(f"Recording class {self.curr_class}...")

        if self.mode == "PREDICTING" and feats is not None and self.predictor is not None:
            self.prediction = self.predictor.predict(feats)
        else:
            self.prediction = -1

        # Debug print léger
        self._t += 1
        if (self._t % 25 == 0) and feats is not None:
            raw = dbg.get("raw", None)
            print("[DBG] raw=", np.round(raw, 3) if raw is not None else None,
                  " feats=", np.round(feats, 3),
                  " pred=", self.prediction)

        # =========================
        # Render overlay (nettoyé)
        # =========================
        disp = frame_1024.copy()
        det = dbg.get("det", None)

        if det is not None:
            face = det.get("face", None)
            iris_L = det.get("iris_L", None)
            iris_R = det.get("iris_R", None)
            nose_c = det.get("nose_c", None)

            if face is not None:
                fx, fy, fw, fh = face
                cv2.rectangle(disp, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

            if nose_c is not None:
                cv2.circle(disp, (int(nose_c[0]), int(nose_c[1])), 4, (0, 0, 255), -1)

            # iris points only (yellow)
            if iris_L is not None:
                cv2.circle(disp, (int(iris_L[0]), int(iris_L[1])), 3, (0, 255, 255), -1)
            if iris_R is not None:
                cv2.circle(disp, (int(iris_R[0]), int(iris_R[1])), 3, (0, 255, 255), -1)

        head_uv = dbg.get("head_uv", None)
        if head_uv is not None:
            cv2.circle(disp, (int(head_uv[0]), int(head_uv[1])), 4, (0, 255, 0), -1)

        status = f"MODE: {self.mode}"
        if self.mode == "RECORDING":
            status += f" | class {self.curr_class} ({self.samples_in_class}/{int(self.spin_samples.value())})"
        if self.mode == "PREDICTING":
            status += f" | Pred: {self.prediction}"
        self.lbl_status.setText(status)

        if feats is None:
            self.lbl_info.setText(f"feats=None ({dbg.get('reason','?')})")
        else:
            dx, dy = dbg.get("delta", (0, 0))
            T = dbg.get("T", np.zeros(3))
            zA, zB, zC = dbg.get("z_mm", (0, 0, 0))
            self.lbl_info.setText(f"δ=({dx:.1f},{dy:.1f})  Tz={T[2]:.2f}cm  zA={zA:.1f}mm")

        self._update_grid_colors()

        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = disp_rgb.shape
        qimg = QImage(disp_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg))


def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1350, 850)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
