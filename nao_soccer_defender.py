
from controller import Robot
import cv2
import numpy as np
import sys
import os
import time
from enum import Enum

# Add current directory to import path for the perception module
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
if current_dir not in sys.path:
    sys.path.append(current_dir)

from perception_vision_module import PerceptionModule, ObjectType, FieldDimensions


class RobotState(Enum):
    SEARCHING_BALL = "searching_ball"  # head scanning only
    DEFENDING = "defending"            # defender strategy execution


class NAOSoccerDefender:
    """Controller dedicated to the Defender role"""

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Fixed role: Defender
        self.role = "DEFENDER"

        # Perception module 
        camera_params = {
            'fx': 600, 'fy': 600, 'cx': 320, 'cy': 240,
            'camera_height': 0.5, 'camera_angle': 0.0
        }
        self.perception = PerceptionModule(camera_params)
        self.field_dims = FieldDimensions()

        # Cameras 
        self.camera_top = self.robot.getDevice("CameraTop")
        self.camera_bottom = self.robot.getDevice("CameraBottom")
        if self.camera_top: self.camera_top.enable(self.timestep)
        if self.camera_bottom: self.camera_bottom.enable(self.timestep)

        # Team color: RED = defend red goal, attack blue goal; BLUE is the opposite
        self.team_color = "RED"  # or "BLUE"

        # Motors
        self.motors = {}
        joint_names = [
            'HeadYaw', 'HeadPitch',
            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
            # Keep arms in a stable posture
            'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
            'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'
        ]
        for j in joint_names:
            m = self.robot.getDevice(j)
            if m: self.motors[j] = m

        self.set_standing_position()
        self.initialize_sensors()

        # State variables
        self.state = RobotState.SEARCHING_BALL
        self.ball_last_seen = None
        self.own_goal_last_seen = None
        self.last_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

        print("NAO Defender initialized.")

    # ---------- Devices / Sensors ----------
    def initialize_sensors(self):
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu: self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice('gps')
        if self.gps: self.gps.enable(self.timestep)

    def set_standing_position(self):
        """Neutral standing pose"""
        pose = {
            'HeadYaw': 0.0, 'HeadPitch': 0.0,
            'LShoulderPitch': 1.2, 'LShoulderRoll': 0.1, 'LElbowYaw': -1.4, 'LElbowRoll': -0.5, 'LWristYaw': 0.0,
            'RShoulderPitch': 1.2, 'RShoulderRoll': -0.1, 'RElbowYaw': 1.4, 'RElbowRoll': 0.5, 'RWristYaw': 0.0,
            'LHipYawPitch': 0.0, 'LHipRoll': 0.0, 'LHipPitch': -0.4, 'LKneePitch': 0.8, 'LAnklePitch': -0.4, 'LAnkleRoll': 0.0,
            'RHipYawPitch': 0.0, 'RHipRoll': 0.0, 'RHipPitch': -0.4, 'RKneePitch': 0.8, 'RAnklePitch': -0.4, 'RAnkleRoll': 0.0
        }
        for j, p in pose.items():
            if j in self.motors:
                self.motors[j].setPosition(p)

    # ---------- Camera ----------
    def get_camera_image(self, use_bottom_camera=False):
        cam = self.camera_bottom if use_bottom_camera else self.camera_top
        if not cam or cam.getWidth() == 0:
            return None
        arr = np.frombuffer(cam.getImage(), dtype=np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    # ---------- Perception ----------
    def process_perception(self):
        img = self.get_camera_image(use_bottom_camera=False)
        if img is None:
            return None

        detections = self.perception.process_frame(img)

        # Ball
        if detections['ball']:
            self.ball_last_seen = {
                'distance': detections['ball'].distance,
                'angle': detections['ball'].angle,
                'pos2d': detections['ball'].position_2d,
                'timestamp': time.time()
            }

        # Own goal
        for goal in detections['goals']:
            if self.team_color == "RED" and goal.object_type == ObjectType.GOAL_RED:
                self.own_goal_last_seen = goal
            if self.team_color == "BLUE" and goal.object_type == ObjectType.GOAL_BLUE:
                self.own_goal_last_seen = goal

        return detections

    # ---------- Simple Locomotion ----------
    def simple_walk(self, forward: float, lateral: float, yaw: float):
        """Small forward/side/yaw adjustments; clamped with dead zones for stability"""
        forward = float(np.clip(forward, -0.2, 0.2))
        lateral = float(np.clip(lateral, -0.15, 0.15))
        yaw = float(np.clip(yaw, -0.5, 0.5))

        hip_pitch = -0.4 + 0.5 * forward
        knee_pitch = 0.8 - 0.6 * forward
        ankle_pitch = -0.4 + 0.25 * forward

        for j, p in [
            ('LHipPitch', hip_pitch), ('LKneePitch', knee_pitch), ('LAnklePitch', ankle_pitch),
            ('RHipPitch', hip_pitch), ('RKneePitch', knee_pitch), ('RAnklePitch', ankle_pitch)
        ]:
            if j in self.motors: self.motors[j].setPosition(p)

        # Use ankle roll to mimic lateral motion (very simplified)
        if 'LAnkleRoll' in self.motors:
            self.motors['LAnkleRoll'].setPosition(np.clip(lateral * 0.5, -0.15, 0.15))
        if 'RAnkleRoll' in self.motors:
            self.motors['RAnkleRoll'].setPosition(np.clip(-lateral * 0.5, -0.15, 0.15))

        if 'HeadYaw' in self.motors:
            self.motors['HeadYaw'].setPosition(yaw)

        self.robot.step(self.timestep)

    def search_for_ball(self):
        """Head scanning to search for the ball"""
        t = time.time()
        yaw = np.sin(t * 0.5) * 1.0
        if 'HeadYaw' in self.motors: self.motors['HeadYaw'].setPosition(yaw)
        if 'HeadPitch' in self.motors: self.motors['HeadPitch'].setPosition(0.2)
        self.robot.step(self.timestep)

    # ---------- Defender ----------
    def defender_action(self):
        now = time.time()
        ball_visible = self.ball_last_seen and (now - self.ball_last_seen['timestamp'] < 2.0)

        # If ball is not visible б· small forward step + scanning
        if not ball_visible:
            self.simple_walk(0.05, 0.0, 0.0)
            self.search_for_ball()
            return

        # Ball and own-goal positions in robot frame (x forward, y to the right)
        b_d = float(self.ball_last_seen['distance'])
        b_a = float(self.ball_last_seen['angle'])
        ball_xy = np.array([b_d * np.cos(b_a), b_d * np.sin(b_a)])

        own_goal_xy = None
        if self.own_goal_last_seen:
            g_d = float(self.own_goal_last_seen.distance)
            g_a = float(self.own_goal_last_seen.angle)
            own_goal_xy = np.array([g_d * np.cos(g_a), g_d * np.sin(g_a)])

        # Direction from ball to own goal (fallback: from ball to robot origin)
        if own_goal_xy is not None:
            v_bg = own_goal_xy - ball_xy
        else:
            v_bg = -ball_xy

        if np.linalg.norm(v_bg) < 1e-3:
            v_bg = np.array([-1.0, 0.0])
        u_bg = v_bg / (np.linalg.norm(v_bg) + 1e-6)

        # Desired standoff behind the ball along u_bg
        standoff = float(np.clip(b_d * 0.5, 0.6, 1.2))
        target_xy = ball_xy + u_bg * standoff

        # Soft constraint: avoid approaching the goal too much
        if own_goal_xy is not None:
            min_goal_dist = 0.8  # meters
            vec = target_xy - own_goal_xy
            d = np.linalg.norm(vec)
            if d < min_goal_dist:
                target_xy = own_goal_xy + vec * (min_goal_dist / (d + 1e-6))

        #  Keep facing the ball
        desired_yaw = np.arctan2(ball_xy[1], ball_xy[0])

        # Proportional control to generate walking command
        r = float(np.linalg.norm(target_xy))
        th = float(np.arctan2(target_xy[1], target_xy[0]))
        vx = np.clip(0.35 * r, -0.18, 0.18)
        vy = np.clip(0.30 * r * np.sin(th), -0.12, 0.12)
        wz = np.clip(0.8 * desired_yaw, -0.4, 0.4)

        # Dead zones to reduce jitter
        if r < 0.1: vx = 0.0
        if abs(th) < 0.05: vy = 0.0
        if abs(desired_yaw) < 0.03: wz = 0.0

        self.simple_walk(vx, vy, wz)
        self.last_cmd = {"vx": vx, "vy": vy, "wz": wz}

    # ---------- State Machine ----------
    def update_state(self, detections):
        if detections and detections['ball']:
            self.state = RobotState.DEFENDING
        else:
            self.state = RobotState.SEARCHING_BALL

    def step_action(self):
        if self.state == RobotState.SEARCHING_BALL:
            self.search_for_ball()
        elif self.state == RobotState.DEFENDING:
            self.defender_action()

    # ---------- Main Loop ----------
    def run(self):
        while self.robot.step(self.timestep) != -1:
            det = self.process_perception()
            self.update_state(det)
            self.step_action()

            # Debug prints (optional)
            if self.ball_last_seen:
                print(f"[Ball] d={self.ball_last_seen['distance']:.2f}m "
                      f"a={np.degrees(self.ball_last_seen['angle']):.1f}бу")
            if self.own_goal_last_seen:
                print(f"[OwnGoal] d={self.own_goal_last_seen.distance:.2f} "
                      f"a={np.degrees(self.own_goal_last_seen.angle):.1f}бу")
            cmd = self.last_cmd
            print(f"[DEF] vx={cmd['vx']:.2f} vy={cmd['vy']:.2f} wz={cmd['wz']:.2f}")


def main():
    NAOSoccerDefender().run()


if __name__ == "__main__":
    main()

