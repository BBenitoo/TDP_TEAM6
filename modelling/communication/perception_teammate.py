# perception/teammate_id.py
import cv2
import numpy as np
# optional: import apriltag if installed
try:
    import apriltag
    APRILTAG_AVAILABLE = True
except Exception:
    APRILTAG_AVAILABLE = False

def detect_apriltags(frame, detector=None):
    if not APRILTAG_AVAILABLE:
        return []
    if detector is None:
        detector = apriltag.Detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    tags = []
    for r in results:
        tags.append({"id": r.tag_id, "center": (int(r.center[0]), int(r.center[1]))})
    return tags

def color_match(frame, hsv_lower, hsv_upper, min_area=200):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    robots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        robots.append({"bbox":(x,y,w,h), "area":area, "center":(x + w//2, y + h//2)})
    return robots

def identify_team(frame, apriltag_detector=None, team_color_bounds=None, comm_observations=None):
    """
    comm_observations: list of dicts from other robots e.g. {'id':'robot_02','pos':(x,y),'timestamp':...}
    Returns list of candidates with possible id and confidence.
    """
    found = []
    # 1) try apriltag
    if APRILTAG_AVAILABLE:
        tags = detect_apriltags(frame, apriltag_detector)
        for t in tags:
            found.append({"method":"apriltag","id":f"tag_{t['id']}","center":t['center'], "conf":0.99})
    # 2) color match
    if team_color_bounds:
        color_robots = color_match(frame, team_color_bounds[0], team_color_bounds[1])
        for r in color_robots:
            found.append({"method":"color","id":None,"center": r['center'], "conf":0.6})
    # 3) fuse with comm observations (nearest neighbor in image->world projection)
    # (Assumes comm_observations already projected to image coords or we have homography)
    if comm_observations:
        # naive: match by proximity in image coords if comm pos projected to pixel coords
        for c in comm_observations:
            # expecting c has 'proj_center' in image coords
            for f in found:
                if f.get("center") and c.get("proj_center"):
                    dx = f['center'][0]-c['proj_center'][0]; dy = f['center'][1]-c['proj_center'][1]
                    if (dx*dx+dy*dy) < 400:  # radius ~20px
                        f['id'] = c['id']
                        f['conf'] = min(0.99, f.get('conf',0.5) + 0.3)
    return found
