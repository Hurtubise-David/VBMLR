import cv2
import mediapipe as mp
import numpy as np
import time

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Coins des yeux (Interne, Externe)
        self.LEFT_EYE_CORNERS = [362, 263] 
        self.RIGHT_EYE_CORNERS = [33, 133]

        # --- PARAMETRES DE SENSIBILITE (A ajuster si besoin) ---
        self.cal_x_min = None
        
        # Sensibilité X : Plus c'est bas, plus c'est sensible (0.04 est très réactif)
        self.sensitivity_x = 0.045
        
        # Sensibilité Y : C'est ici le secret pour le haut/bas. 
        # C'est très bas car le mouvement vertical normalisé est minuscule.
        self.sensitivity_y = 0.025 

        # Facteur de lissage (Plus grand = plus fluide mais un peu de latence)
        self.history_len = 6 
        self.history_x = []
        self.history_y = []

    def get_landmark_point(self, landmarks, idx, w, h):
        return np.array([int(landmarks[idx].x * w), int(landmarks[idx].y * h)])

    def get_relative_gaze(self, eye_corners, iris_center):
        """
        NOUVELLE MATHEMATIQUE :
        Au lieu d'utiliser la hauteur de l'oeil (instable à cause des paupières),
        on utilise la largeur de l'oeil comme référence pour X et Y.
        """
        # eye_corners = [Coin Interne, Coin Externe]
        
        # 1. Calcul du centre géométrique de l'oeil (le point milieu entre les coins)
        eye_center_x = (eye_corners[0][0] + eye_corners[1][0]) / 2
        eye_center_y = (eye_corners[0][1] + eye_corners[1][1]) / 2
        
        # 2. Largeur de l'oeil (C'est notre métrique stable)
        eye_width = np.linalg.norm(eye_corners[1] - eye_corners[0])
        if eye_width == 0: eye_width = 1

        # 3. Déplacement de l'iris par rapport au centre, normalisé par la largeur
        # dx positif = regarde à droite (pour l'oeil gauche)
        # dy positif = regarde en bas
        dx = (iris_center[0] - eye_center_x) / eye_width
        dy = (iris_center[1] - eye_center_y) / eye_width
        
        return dx, dy

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        gaze_x, gaze_y = None, None

        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            
            # --- Récupération des points ---
            p_l_pts = [self.get_landmark_point(mesh_points, idx, w, h) for idx in self.LEFT_EYE_CORNERS]
            p_iris_l = self.get_landmark_point(mesh_points, 473, w, h)
            
            p_r_pts = [self.get_landmark_point(mesh_points, idx, w, h) for idx in self.RIGHT_EYE_CORNERS]
            p_iris_r = self.get_landmark_point(mesh_points, 468, w, h)
            
            # --- Calcul des Déplacements Relatifs ---
            dx_l, dy_l = self.get_relative_gaze(p_l_pts, p_iris_l)
            dx_r, dy_r = self.get_relative_gaze(p_r_pts, p_iris_r)

            # Moyenne des deux yeux
            avg_dx = (dx_l + dx_r) / 2
            avg_dy = (dy_l + dy_r) / 2

            # --- AUTO-CALIBRATION CONTINUE ---
            # On considère la position actuelle comme le "Centre" au démarrage
            if self.cal_x_min is None:
                self.center_x = avg_dx
                self.center_y = avg_dy
                self.cal_x_min = 0 # Dummy value to say "initialized"

            # Calcul de la distance par rapport au centre calibré
            delta_x = avg_dx - self.center_x
            delta_y = avg_dy - self.center_y

            # --- MAPPING VERS ECRAN (0.0 à 1.0) ---
            # On utilise une fonction sigmoïde simplifiée (clamping linéaire)
            # -sensitivity -> 0.0
            # 0 -> 0.5
            # +sensitivity -> 1.0
            
            norm_x = 0.5 + (delta_x / (2 * self.sensitivity_x))
            norm_y = 0.5 + (delta_y / (2 * self.sensitivity_y))

            # Clamping pour ne pas sortir de l'écran
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            # --- LISSAGE ---
            self.history_x.append(norm_x)
            self.history_y.append(norm_y)
            if len(self.history_x) > self.history_len:
                self.history_x.pop(0)
                self.history_y.pop(0)
            
            gaze_x = sum(self.history_x) / len(self.history_x)
            gaze_y = sum(self.history_y) / len(self.history_y)

            # Debug visuel : Ligne entre les coins des yeux pour visualiser l'axe
            cv2.line(frame, tuple(p_l_pts[0]), tuple(p_l_pts[1]), (255, 0, 0), 1)
            cv2.line(frame, tuple(p_r_pts[0]), tuple(p_r_pts[1]), (255, 0, 0), 1)
            cv2.circle(frame, tuple(p_iris_l), 2, (0, 255, 255), -1)

        return frame, gaze_x, gaze_y

    def reset_calibration(self):
        """Force le recalibrage sur la position actuelle"""
        self.cal_x_min = None

def draw_ui(screen_img, gaze_x, gaze_y):
    h, w, _ = screen_img.shape
    rows, cols = 3, 3
    active_idx = -1
    
    if gaze_x is not None:
        col = int(gaze_x * cols)
        row = int(gaze_y * rows)
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        active_idx = row * cols + col + 1

    # Grille
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j + 1
            x1, y1 = j * (w//cols), i * (h//rows)
            x2, y2 = x1 + (w//cols), y1 + (h//rows)
            
            color = (100,100,100)
            if idx == active_idx:
                cv2.rectangle(screen_img, (x1, y1), (x2, y2), (0, 200, 0), -1)
                color = (0,0,0)
            
            cv2.rectangle(screen_img, (x1, y1), (x2, y2), (50, 50, 50), 2)
            cv2.putText(screen_img, str(idx), (x1+int(w/cols/2)-20, y1+int(h/rows/2)+20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    if gaze_x is not None:
        cv2.circle(screen_img, (int(gaze_x * w), int(gaze_y * h)), 20, (0, 0, 255), -1)

    return active_idx

def main():
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        frame_debug, gx, gy = tracker.process_frame(frame)
        
        ui_screen = np.zeros((600, 800, 3), dtype=np.uint8)
        active_cell = draw_ui(ui_screen, gx, gy)
        
        # Petit retour caméra
        small = cv2.resize(frame_debug, (200, 150))
        ui_screen[0:150, 0:200] = small
        
        cv2.putText(ui_screen, "REGARDEZ LE CENTRE + Appuyez sur 'C'", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Gaze Tracker V2", ui_screen)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord('c'): 
            tracker.reset_calibration()
            print("Calibration Reset sur le centre.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()