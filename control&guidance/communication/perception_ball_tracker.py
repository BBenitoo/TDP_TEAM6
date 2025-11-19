# perception/ball_tracker.py
import cv2
import numpy as np
import time

class SimpleKalman2D:
    def __init__(self, dt=0.05, process_var=1.0, meas_var=5.0):
        # state: [x, y, vx, vy]
        self.dt = dt
        self.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.P = np.eye(4) * 500.
        self.x = np.zeros((4,1))
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(2,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

class BallDetector:
    def __init__(self, hsv_lower=(5,100,100), hsv_upper=(20,255,255), min_area=200, homography=None):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area
        self.kf = SimpleKalman2D()
        self.last_detection_time = None
        self.homography = homography  # 3x3 matrix to convert image->field coords if available

    def adaptive_calibrate(self, sample_frame):
        # auto-calc HSV range from sample ROI: placeholder for manual or auto calibration UI
        hsv = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2HSV)
        # compute hist or use median values: this is a simple example
        med = np.median(hsv.reshape(-1,3), axis=0)
        h, s, v = med.astype(int)
        self.hsv_lower = np.array([max(h-15,0), max(s-60,40), max(v-80,40)])
        self.hsv_upper = np.array([min(h+15,179), 255, 255])

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # morphology
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area: continue
            (x,y), r = cv2.minEnclosingCircle(c)
            if area > best_area:
                best = (int(x), int(y), int(r), area)
                best_area = area
        if best:
            x,y,r,area = best
            # convert to field coords if homography provided
            img_pt = np.array([ [x,y,1] ]).T
            if self.homography is not None:
                field_pt = self.homography @ img_pt
                field_pt = field_pt/field_pt[2]
                pos = (float(field_pt[0]), float(field_pt[1]))
            else:
                pos = (x,y)
            # update kf
            self.kf.predict()
            self.kf.update(pos)
            self.last_detection_time = time.time()
            state = self.kf.get_state()
            # confidence heuristic: area / (area + const)
            confidence = min(0.99, area / (area + 1000.0))
            return {"pos": (state[0], state[1]), "vel": (state[2], state[3]), "conf": confidence, "raw": pos}
        else:
            # no detection -> predict
            self.kf.predict()
            state = self.kf.get_state()
            return {"pos": (state[0], state[1]), "vel": (state[2], state[3]), "conf": 0.0, "raw": None}
