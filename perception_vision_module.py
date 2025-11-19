

import cv2
import numpy as np
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from calibration_module import CameraCalibration, CoordinateTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectType(Enum):
    """Types of objects that can be detected"""
    BALL = "ball"
    GOAL_POST = "goal_post"
    GOAL_BLUE = "goal_blue"
    GOAL_RED = "goal_red"
    FIELD_LINE = "field_line"
    CENTER_CIRCLE = "center_circle"
    PENALTY_MARK = "penalty_mark"
    CORNER = "corner"
    T_JUNCTION = "t_junction"
    L_JUNCTION = "l_junction"
    CROSS = "cross"

@dataclass
class DetectedObject:
    """Represents a detected object in the field"""
    object_type: ObjectType
    position_2d: Tuple[int, int]  # Pixel coordinates (x, y)
    position_3d: Optional[Tuple[float, float, float]] = None  # World coordinates (x, y, z)
    confidence: float = 0.0
    size: Optional[Tuple[int, int]] = None  # Width, Height in pixels
    distance: Optional[float] = None  # Distance in meters
    angle: Optional[float] = None  # Angle in radians

@dataclass
class RobotPose:
    """Robot's position and orientation estimate"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    confidence: float = 0.0

class FieldDimensions:
 
    FIELD_LENGTH = 9.0
    FIELD_WIDTH = 6.0
    GOAL_WIDTH = 2.6
    GOAL_HEIGHT = 1.2
    GOAL_AREA_LENGTH = 1.0
    GOAL_AREA_WIDTH = 3.0
    PENALTY_AREA_LENGTH = 2.0
    PENALTY_AREA_WIDTH = 5.0
    CENTER_CIRCLE_RADIUS = 0.75
    PENALTY_MARK_DISTANCE = 1.5
    LINE_WIDTH = 0.05
    BALL_DIAMETER = 0.10  # FIFA size 1 ball

class ColorRanges:
    """HSV color ranges for object detection"""
    # Ball (dark and white pattern) - detect white parts
    BALL_WHITE_LOWER = np.array([0, 0, 200])
    BALL_WHITE_UPPER = np.array([180, 30, 255])
    BALL_BLACK_LOWER = np.array([0, 0, 0])
    BALL_BLACK_UPPER = np.array([180, 255, 50])
    
    # Field (green)
    FIELD_GREEN_LOWER = np.array([40, 50, 50])
    FIELD_GREEN_UPPER = np.array([80, 255, 255])
    
    # Lines (white)
    LINE_WHITE_LOWER = np.array([0, 0, 200])
    LINE_WHITE_UPPER = np.array([180, 30, 255])
    
    # Goal posts (white)
    GOAL_WHITE_LOWER = np.array([0, 0, 200])
    GOAL_WHITE_UPPER = np.array([180, 30, 255])
    
    # Team colors (red / blue markers)
    RED_LOWER = np.array([0, 100, 100])
    RED_UPPER = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])
    RED_UPPER2 = np.array([180, 255, 255])
    BLUE_LOWER = np.array([100, 150, 50])
    BLUE_UPPER = np.array([130, 255, 255])

class PerceptionModule:
    """Main perception module for detecting objects and estimating robot pose"""
    
    def __init__(self, camera_params: Dict = None):
       
        self.field_dims = FieldDimensions()
        self.color_ranges = ColorRanges()
        
        # Camera parameters (default NAO camera specs)
        if camera_params:
            self.camera_params = camera_params
        else:
            self.camera_params = {
                'fx': 600,  # Focal length x
                'fy': 600,  # Focal length y
                'cx': 320,  # Principal point x
                'cy': 240,  # Principal point y
                'camera_height': 0.5,  # meters
                'camera_angle': 0.0  # radians 
            }
        
        # Initialize calibration and coordinate transformer
        try:
            self._calib = CameraCalibration()
            self._coord = CoordinateTransformer(self._calib)
        except Exception as e:
            logger.warning(f"Calibration init failed: {e}")
            self._calib = None
            self._coord = None
        
        # Robot pose estimate
        self.robot_pose = RobotPose()
        
        # Detection history for filtering
        self.detection_history = []
        self.max_history = 5
        
        logger.info("Perception module initialized")
    
    def process_frame(self, image: np.ndarray) -> Dict:
        
        # Undistort image if calibration available
        if hasattr(self, '_calib') and self._calib is not None:
            try:
                image = self._calib.undistort_image(image, use_bottom_camera=False)
            except Exception as e:
                logger.debug(f"Undistort failed: {e}")
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect objects
        ball = self.detect_ball(image, hsv)
        goals = self.detect_goals(image, hsv)
        lines = self.detect_field_lines(image, hsv)
        landmarks = self.detect_landmarks(lines)
        
        # Update localization
        self.update_localization(ball, goals, landmarks)
        
        # Compile results
        results = {
            'ball': ball,
            'goals': goals,
            'lines': lines,
            'landmarks': [lm for lm in landmarks if lm is not None],
            'robot_pose': self.robot_pose
        }
        
        return results
    
    def detect_ball(self, image: np.ndarray, hsv: np.ndarray) -> Optional[DetectedObject]:
      
        # Emphasize white regions (ball patches)
        mask_white = cv2.inRange(hsv, self.color_ranges.BALL_WHITE_LOWER, 
                                 self.color_ranges.BALL_WHITE_UPPER)
        kernel = np.ones((5, 5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_score = 0.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            
            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            # Check for dark pattern inside white region
            mask_cnt = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask_cnt)
            # lower V/S suggests pattern
            pattern_score = 1.0 - (mean_val[1] / 255.0)
            
            # Combined score
            score = 0.5 * circularity + 0.2 * pattern_score + 0.3 * \
                    (1.0 - abs(aspect_ratio - 1.0))
            
            if score < 0.4:
                continue
            
            # Evaluate black ratio inside
            mask_black = cv2.inRange(hsv[y:y+h, x:x+w], 
                                     self.color_ranges.BALL_BLACK_LOWER,
                                     self.color_ranges.BALL_BLACK_UPPER)
            black_ratio = np.sum(mask_black > 0) / (w*h + 1e-6)
            if black_ratio < 0.1 or black_ratio > 0.6:
                continue
            
            # Calculate score
            score = circularity * (1 - abs(aspect_ratio - 1)) * \
                   (1 - abs(black_ratio - 0.3))
            
            if score > max_score:
                max_score = score
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate distance based on ball size
                ball_pixel_diameter = (w + h) / 2
                distance = self.estimate_ball_distance(ball_pixel_diameter)
                angle = self.pixel_to_angle(center_x)
                
                best_ball = DetectedObject(
                    object_type=ObjectType.BALL,
                    position_2d=(center_x, center_y),
                    confidence=min(score, 1.0),
                    size=(w, h),
                    distance=distance,
                    angle=angle
                )
        # Enhance with ground projection if calibration is available
        try:
            if hasattr(self, '_coord') and self._coord is not None and best_ball is not None:
                gxgy = self._coord.estimate_ground_position(center_x, center_y, image.shape[1], image.shape[0], use_bottom_camera=False)
                if gxgy is not None:
                    gx, gy = gxgy
                    best_ball.position_3d = (float(gx), float(gy), 0.0)
                    best_ball.distance = float(np.hypot(gx, gy))
        except Exception as e:
            logger.debug(f"Ground projection failed: {e}")
        
        return best_ball
    
    def detect_goals(self, image: np.ndarray, hsv: np.ndarray) -> List[DetectedObject]:
       
        goals = []
        
        # Detect white regions (goal posts)
        mask_white = cv2.inRange(hsv, self.color_ranges.GOAL_WHITE_LOWER,
                                 self.color_ranges.GOAL_WHITE_UPPER)
        kernel = np.ones((5, 5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(mask_white, 50, 150)
        
        # Hough lines to detect vertical structures
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Group lines into potential goal posts
            vertical_lines = []
            h_img, w_img = image.shape[:2]
            left_border = int(0.05 * w_img)
            right_border = int(0.95 * w_img)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Ensure y1 is top, y2 is bottom
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                dx = x2 - x1; dy = y2 - y1
                angle = np.abs(np.arctan2(dy, dx))
                height = abs(y2 - y1)
                x_avg = (x1 + x2) / 2
                # Filters: near-vertical, sufficiently tall, not on image borders, top half origin
                if angle > np.pi * 5/12 and height > 0.2 * h_img and left_border < x_avg < right_border and y1 < 0.8 * h_img:
                    vertical_lines.append([int(x1), int(y1), int(x2), int(y2)])
            
            # Pair vertical lines as goal posts
            if len(vertical_lines) >= 2:
                # Sort by x-coordinate
                vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
                
                # Check for goal post pairs
                for i in range(len(vertical_lines) - 1):
                    line1 = vertical_lines[i]
                    line2 = vertical_lines[i + 1]
                    
                    x1_avg = (line1[0] + line1[2]) / 2
                    x2_avg = (line2[0] + line2[2]) / 2
                    
                    # Check if distance between posts is reasonable
                    pixel_distance = abs(x2_avg - x1_avg)
                    
                    # Estimate if this could be a goal based on expected size and geometry
                    h1 = abs(line1[3]-line1[1]); h2 = abs(line2[3]-line2[1])
                    height_ratio = min(h1, h2) / max(h1, h2)
                    y_top1 = min(line1[1], line1[3]); y_top2 = min(line2[1], line2[3])
                    top_diff = abs(y_top1 - y_top2)
                    vertical_overlap = min(max(line1[1], line1[3]), max(line2[1], line2[3])) - max(min(line1[1], line1[3]), min(line2[1], line2[3]))
                    if pixel_distance > 30 and pixel_distance < 0.8 * w_img and height_ratio > 0.6 and top_diff < 0.15 * h_img and vertical_overlap > 0.2 * h_img:
                        center_x = int((x1_avg + x2_avg) / 2)
                        center_y = int((min(line1[1], line1[3], line2[1], line2[3]) +
                                      max(line1[1], line1[3], line2[1], line2[3])) / 2)
                        
                        # Determine goal color (team)
                        goal_type = self.identify_goal_team(image, hsv, 
                                                           (center_x, center_y))
                        
                        # Estimate distance
                        height = max(abs(line1[3] - line1[1]), 
                                   abs(line2[3] - line2[1]))
                        distance = self.estimate_goal_distance(height)
                        angle = self.pixel_to_angle(center_x)
                        
                        goals.append(DetectedObject(
                            object_type=goal_type,
                            position_2d=(center_x, center_y),
                            confidence=0.7,
                            size=(int(pixel_distance), int(height)),
                            distance=distance,
                            angle=angle
                        ))
        
        return goals
    
    def detect_field_lines(self, image: np.ndarray, hsv: np.ndarray) -> List[np.ndarray]:
       
        # Create mask for white lines
        mask_white = cv2.inRange(hsv, self.color_ranges.LINE_WHITE_LOWER,
                                 self.color_ranges.LINE_WHITE_UPPER)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        # Edge detection
        edges = cv2.Canny(mask_white, 50, 150)
        
        # Hough transform for line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [line[0] for line in lines]
    
    def detect_landmarks(self, lines: List[np.ndarray]) -> List[Optional[DetectedObject]]:
    
        if not lines:
            return []
        
        # Find intersections
        intersections = self.find_line_intersections(lines)
        
        landmarks = []
        
        for inter_point, intersecting_lines in intersections:
            # Classify based on number of intersecting lines
            num_lines = len(intersecting_lines)
            landmark_type = None
            confidence = 0.5
            
            if num_lines == 2:
                # L or T junction depending on angle
                angle = self.angle_between_lines(intersecting_lines[0], intersecting_lines[1])
                if abs(angle - np.pi/2) < np.pi/6:
                    landmark_type = ObjectType.L_JUNCTION
                    confidence = 0.7
                else:
                    landmark_type = ObjectType.T_JUNCTION
                    confidence = 0.6
            elif num_lines == 3:
                # T-junction
                landmark_type = ObjectType.T_JUNCTION
                confidence = 0.6
            elif num_lines >= 4:
                # Cross or center point
                landmark_type = ObjectType.CROSS
                confidence = 0.5
            
            if landmark_type:
                # Estimate distance to landmark
                distance = self.estimate_landmark_distance(inter_point[1])
                angle = self.pixel_to_angle(inter_point[0])
                
                landmarks.append(DetectedObject(
                    object_type=landmark_type,
                    position_2d=inter_point,
                    confidence=confidence,
                    distance=distance,
                    angle=angle
                ))
        
        return landmarks
    
    def estimate_ball_distance(self, ball_pixel_diameter: float) -> float:
        """Estimate ball distance based on its pixel size"""
        # Simple pinhole approximation with assumed ball size
        if ball_pixel_diameter <= 0:
            return 10.0
        focal = self.camera_params['fx']  # assume square pixels
        distance = (focal * self.field_dims.BALL_DIAMETER) / (ball_pixel_diameter + 1e-6)
        return max(0.1, min(12.0, distance))
    
    def estimate_goal_distance(self, goal_pixel_height: float) -> float:
        """Estimate goal distance based on its pixel height"""
        if goal_pixel_height <= 0:
            return 10.0
        focal = self.camera_params['fy']
        distance = (focal * self.field_dims.GOAL_HEIGHT) / (goal_pixel_height + 1e-6)
        return max(0.3, min(20.0, distance))
    
    def estimate_landmark_distance(self, pixel_y: int, image_height: int = 480) -> float:
        """Rough distance estimate based on y position (heuristic)"""
        horizon_y = int(image_height * 0.3)  # approximate horizon line
        
        if pixel_y > horizon_y:
            # Below horizon - closer
            normalized_y = (pixel_y - horizon_y) / (image_height - horizon_y)
            distance = 0.5 + normalized_y * 5.0  # 0.5m to 5.5m range
        else:
            # Above horizon - farther
            normalized_y = (horizon_y - pixel_y) / horizon_y
            distance = 5.5 + normalized_y * 5.0  # 5.5m to 10.5m range
        
        return min(max(distance, 0.5), 12.0)
    
    def pixel_to_angle(self, pixel_x: int) -> float:
   
        fx = float(self.camera_params.get('fx', 600))
        cx = float(self.camera_params.get('cx', 320))
        # atan2 provides correct sign and is stable for small denominators
        return math.atan2((pixel_x - cx), fx)
    
    def visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
   
        output = image.copy()
        
        # Draw ball
        if detections['ball']:
            ball = detections['ball']
            cv2.circle(output, ball.position_2d, 10, (0, 255, 255), 2)
            cv2.putText(output, f"Ball d={ball.distance:.2f}m a={np.degrees(ball.angle):.1f}deg",
                        (ball.position_2d[0] + 10, ball.position_2d[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw goals
        for goal in detections['goals']:
            color = (255, 0, 0) if goal.object_type == ObjectType.GOAL_BLUE else (0, 0, 255)
            cx, cy = goal.position_2d
            cv2.rectangle(output, (cx - goal.size[0]//2, cy - goal.size[1]//2),
                          (cx + goal.size[0]//2, cy + goal.size[1]//2), color, 2)
            cv2.putText(output, f"{goal.object_type.value} d={goal.distance:.2f}",
                        (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw landmarks
        for lm in detections['landmarks']:
            if lm is None:
                continue
            cv2.circle(output, (int(lm.position_2d[0]), int(lm.position_2d[1])), 5, (0, 255, 0), -1)
            cv2.putText(output, f"{lm.object_type.value}", (int(lm.position_2d[0]) + 5, int(lm.position_2d[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return output
    
    # ---------- Helper geometry / intersection methods (unchanged) ----------
    def find_line_intersections(self, lines: List[np.ndarray], threshold: float = 10.0) -> List[Tuple[Tuple[int, int], List[np.ndarray]]]:
   
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                point = self.line_intersection(lines[i], lines[j])
                if point is not None:
                    # Check if near existing intersection
                    found = False
                    for k, (existing_point, existing_lines) in enumerate(intersections):
                        dist = np.sqrt((point[0] - existing_point[0])**2 + 
                                     (point[1] - existing_point[1])**2)
                        if dist < threshold:
                            # Merge with existing
                            intersections[k] = ((
                                (existing_point[0] + point[0]) // 2, 
                                (existing_point[1] + point[1]) // 2
                            ), existing_lines + [lines[i], lines[j]])
                            found = True
                            break
                    
                    if not found:
                        intersections.append((point, [lines[i], lines[j]]))
        
        return intersections
    
    def line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[int, int]]:
    
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        
        # Check if within segment bounds
        if (min(x1, x2) - 1 <= px <= max(x1, x2) + 1 and
            min(y1, y2) - 1 <= py <= max(y1, y2) + 1 and
            min(x3, x4) - 1 <= px <= max(x3, x4) + 1 and
            min(y3, y4) - 1 <= py <= max(y3, y4) + 1):
            return (int(px), int(py))
        
        return None
    
    def angle_between_lines(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """Calculate angle between two lines in radians"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x4 - x3, y4 - y3])
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm < 1e-6:
            return 0.0
        
        cos_theta = max(min(dot / norm, 1.0), -1.0)
        return np.arccos(cos_theta)


    
