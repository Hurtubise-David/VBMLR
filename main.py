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
from centeriris import CenterIrisCpp
from typing import Tuple

# ============================================================
# 1) CONFIG
# ============================================================
class Config:
    CAM_W, CAM_H = 640, 480
    W_RESIZE, H_RESIZE = 1024, 768

    # Intrinsics (TO DO need to create a calibrator and charge .yml)
    FX, FY = 600.0, 600.0
    CX, CY = 320.0, 240.0

    # Blur scale (sigma0) add know blur
    SIGMA0 = 2.0

    # blur->depth (Eq. 9 from paper)
    F_MM = 4.033      # F (focal length) mm
    f_MM = 4.0676     # f (distance lense->image plane) mm
    F_NUM = 2.8       # f-number
    K_CAL = 0.0013    # constant k
    U_MM = 500.0      # u (perfect focus position)

    NB_CLASSES = 9

    # ROI nose (640x480) : center box to 100x100
    NOSE_ROI_W, NOSE_ROI_H = 120, 120
    NOSE_ROI_X = (CAM_W - NOSE_ROI_W) // 2
    NOSE_ROI_Y = (CAM_H - NOSE_ROI_H) // 2

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
# 2) I/O Tools
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
# 5) Blur -> sigma map -> depth -> Head pose + iris displacement
# ============================================================
class BlurFeatureExtractor:
    def __init__(self):
        self.ci = CenterIrisCpp()

        nose_path = Config.cascade_path("haarcascade_mcs_nose.xml")
        self.nose_cascade = cv2.CascadeClassifier(nose_path)
        if self.nose_cascade.empty():
            self.nose_cascade = None

    @staticmethod
    def blur_sigma_map(frame_bgr_640: np.ndarray,
                        sigma: float = 1.0,
                        lambd: float = 0.003) -> np.ndarray:
        """
        From paper (Eq. 15):
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
        # img1BW = gray (float32), img2BW = GaussianBlur(img1BW, sigma)
        gray = cv2.cvtColor(frame_bgr_640, cv2.COLOR_BGR2GRAY).astype(np.float32)

        k = int(4 * sigma + 1)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

        # img1BW = Sobelx^2 + Sobely^2  (no sqrt)
        gx1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        gy1 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        g1 = gx1 * gx1 + gy1 * gy1

        gx2 = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        gy2 = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        g2 = gx2 * gx2 + gy2 * gy2

        # Avoid div0 like C++ checks
        eps = 1e-12
        R = np.sqrt(np.maximum(g1, 0.0) / (np.maximum(g2, eps)))

        # if R <= 1 and R > 0: val = 1 - exp(-lambda/(R^2)), R = val
        # if R == 0: R = 0
        m = (R > 0) & (R <= 1.0)
        Rm = R[m]
        R[m] = 1.0 - np.exp(-lambd / (Rm * Rm + eps))

        # R = R - 1 ; R = sqrt(R) ; sigmas = sigma / R
        R = R - 1.0
        R = np.sqrt(np.maximum(R, eps))
        sigmas = sigma / R
        return sigmas.astype(np.float32)

    @staticmethod
    def depth_from_sigma_mm(sigma):
        """
        From paper Eq. (9) (blur->depth) mm:
          z = (F f) / (f - F ± k F N σ)
        Sign depend z>=u or not.
        More stable version:
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
    
    def blur_head_pose_estimation(self, sigmas_640, nose_bbox, head_center_uv_640):
        from typing import Tuple
        x1, y1, x2, y2 = nose_bbox  # absolute coords in 640x480

        r = sigmas_640[y1:y2, x1:x2]
        if r.size == 0:
            return None, {"reason": "empty_nose_roi", "nose_bbox": nose_bbox}
        sigNose = float(np.mean(r))

        fx, fy = float(Config.FX), float(Config.FY)
        cx, cy = float(Config.CX), float(Config.CY)

        f = float(Config.F_MM)
        v = float(Config.f_MM)
        k = float(Config.K_CAL)

        nose_Z = (f * v) / ((v - f) + (k * (f / 2.0) * sigNose))

        centerNose_x = 0.5 * (x1 + x2)
        centerNose_y = 0.5 * (y1 + y2)

        nose_X = (nose_Z / fx) * (centerNose_x - cx)
        nose_Y = (nose_Z / -fy) * (centerNose_y - cy)

        ref_u, ref_v = head_center_uv_640
        ref_Z = nose_Z + 150.0
        ref_X = (ref_Z / fx) * (ref_u - cx)
        ref_Y = (ref_Z / -fy) * (ref_v - cy)

        R = np.array([nose_X - ref_X, nose_Y - ref_Y, nose_Z - ref_Z], dtype=np.float64)
        n = float(np.linalg.norm(R)) + 1e-12
        R = R / n

        # Return LIST (headpose6 + [dx,dy])
        headpose6 = [float(ref_X), float(ref_Y), float(ref_Z), float(R[0]), float(R[1]), float(R[2])]

        dbg = {
            "sigNose": sigNose,
            "nose_Z": float(nose_Z),
            "nose_X": float(nose_X),
            "nose_Y": float(nose_Y),
            "ref_XYZ": (float(ref_X), float(ref_Y), float(ref_Z)),
            "R": (float(R[0]), float(R[1]), float(R[2])),
            "nose_bbox": nose_bbox,
        }
        return headpose6, dbg

    
    def detect_nose_bbox_center_roi(self, frame_bgr_640):
        if self.nose_cascade is None:
            return None

        x0, y0 = Config.NOSE_ROI_X, Config.NOSE_ROI_Y
        roi = frame_bgr_640[y0:y0+Config.NOSE_ROI_H, x0:x0+Config.NOSE_ROI_W]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        noses = self.nose_cascade.detectMultiScale(
            gray, 1.2, 3,
            flags=cv2.CASCADE_DO_CANNY_PRUNING,
            minSize=(25, 25)
        )
        if len(noses) == 0:
            return None

        # choose ROI center 100x100
        best = None
        best_d = 1e18
        for (x, y, w, h) in noses:
            cx = x + w/2.0
            cy = y + h/2.0
            d = (cx-50.0)**2 + (cy-50.0)**2
            if d < best_d:
                best_d = d
                x1 = x0 + int(x)
                y1 = y0 + int(y)
                x2 = x1 + int(w)
                y2 = y1 + int(h)
                best = (x1, y1, x2, y2)
        return best
    

    def extract_features(self, frame_bgr_640):
        # 1) CenterIris (1024x768)
        ci_1024, frame_1024 = self.ci.center_iris(frame_bgr_640)
        if ci_1024 is None:
            return None, frame_1024, {"reason": "no_face_or_iris"}

        # 2) Nose bbox on 640 (fixed ROI)
        nose_bbox = self.detect_nose_bbox_center_roi(frame_bgr_640)
        if nose_bbox is None:
            return None, frame_1024, {"reason": "no_nose", "ci": ci_1024}

        # 3) Sigma map on 640
        sigmas = self.blur_sigma_map(frame_bgr_640) 

        # 4) Head center: face center (1024) to coords 640
        ix, iy, fx, fy, fw, fh = ci_1024
        headCenter_1024_u = (fx + (fx + fw)) / 2.0
        headCenter_1024_v = (fy + (fy + fh)) / 2.0
        headCenter2_u = headCenter_1024_u * (Config.CAM_W / Config.W_RESIZE)
        headCenter2_v = headCenter_1024_v * (Config.CAM_H / Config.H_RESIZE)

        # 5) Headpose from blur
        headpose6, dbg_hp = self.blur_head_pose_estimation(sigmas, nose_bbox, (headCenter2_u, headCenter2_v))
        if headpose6 is None:
            return None, frame_1024, {"reason": "headpose_fail", "ci": ci_1024, "nose_bbox": nose_bbox}

        # 6) Disparity iris - face_center
        dx = ix - headCenter_1024_u
        dy = iy - headCenter_1024_v

        feats8_raw = np.array(headpose6 + [dx/100.0, dy/100.0], dtype=np.float64)
        feats8 = self.normalize_plus_one(feats8_raw)

        dbg = {
            "ci": ci_1024,
            "nose_bbox": nose_bbox,
            "sigmas": sigmas,
            "headpose6": headpose6,
            "delta": (dx, dy),
            "raw": feats8_raw,
            "feats": feats8,
            "hp_dbg": dbg_hp,
        }
        return feats8, frame_1024, dbg



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
            raise RuntimeError(f"Not enough samples for class{c}: {p}")
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
        self.setWindowTitle("VBMLR - Eye Gaze Prediction")

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
            QMessageBox.warning(self, "Failed", "Impossible to read camera_matrix from this .yml")

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
            QMessageBox.warning(self, "Dataset missing", "Select first a dataset folder.")
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
            QMessageBox.warning(self, "Dataset missing", "Select first a dataset folder.")
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
        
        ci = dbg.get("ci", None)
        if ci is not None:
            ix, iy, fx, fy, fw, fh = ci
            cv2.circle(disp, (ix, iy), 4, (0,255,255), -1)
            cv2.rectangle(disp, (fx, fy), (fx+fw, fy+fh), (255,0,0), 2)

        nb = dbg.get("nose_bbox", None)
        if nb is not None:
            x, y, w, h = nb
            sx = Config.W_RESIZE / Config.CAM_W
            sy = Config.H_RESIZE / Config.CAM_H
            x = int(x * sx); y = int(y * sy)
            w = int(w * sx); h = int(h * sy)
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0,0,255), 2)        

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

        # Debug print
        self._t += 1
        if (self._t % 25 == 0) and feats is not None:
            raw = dbg.get("raw", None)
            print("[DBG] raw=", np.round(raw, 3) if raw is not None else None,
                  " feats=", np.round(feats, 3),
                  " pred=", self.prediction)

        # =========================
        # Render overlay
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
