import cv2
import numpy as np
from ultralytics import YOLO
from config import Config
import time
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import math
import supervision as sv

class TrafficViolationDetector:
    def __init__(self):
        self.config = Config()
        self.vehicle_model = YOLO(self.config.YOLO_MODEL)
        self.helmet_model = YOLO(self.config.HELMET_MODEL)
  # Khởi tạo ByteTrack tracker (support multiple supervision versions)
        try:
            # Preferred/common signature
            self.tracker = sv.ByteTrack(
                track_thresh=0.2,   # ngưỡng confidence để track
                match_thresh=0.8,    # ngưỡng IOU matching
                frame_rate=self.config.FPS
            )
        except TypeError:
            try:
                # Alternate signatures seen in some versions
                self.tracker = sv.ByteTrack(
                     track_thresh=0.20,    # ngưỡng thấp hơn để không bỏ lỡ xe nhỏ xa
                     match_thresh=0.9,     # yêu cầu IOU cao để tránh gán nhầm
                     track_buffer=90,      # giữ ID nếu YOLO mất detection tạm thời
                     frame_rate=self.config.FPS
)
            except TypeError:
                try:
                    # Minimal constructor fallback
                    self.tracker = sv.ByteTrack()
                except Exception as e:
                    # If ByteTrack cannot be constructed, set to None and continue (tracking disabled)
                    print(f"Warning: could not initialize ByteTrack tracker: {e}")
                    self.tracker = None
        # Track vehicles across frames: each track stores bbox, centroid, missing_frames and position history
        self.vehicle_tracks = {}  # track_id -> { 'bbox':..., 'centroid':(...), 'missing':int, 'positions':deque }
        self.track_id_counter = 0

        # Violation tracking
        self.violations = []
        self.violation_history = deque(maxlen=100)

        # Traffic light state
        self.traffic_light_state = 'unknown'

        # Speed calculation kept per-track inside vehicle_tracks
        
    def detect_vehicles(self, frame):
        """Detect vehicles in frame using YOLO"""
        # Use very low confidence threshold to catch motorcycles
        min_conf = min(0.25, self.config.MOTORCYCLE_CONFIDENCE_THRESHOLD)

        # Tăng kích thước ảnh đầu vào để YOLO có thêm chi tiết
        results = self.vehicle_model(frame, conf=0.25, imgsz=960)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Filter for vehicle classes only
                    if class_id in self.config.VEHICLE_CLASSES:
                        # Use different thresholds for different vehicle types
                        if class_id == self.config.MOTORCYCLE_CLASS_ID:
                            if confidence < self.config.MOTORCYCLE_CONFIDENCE_THRESHOLD:
                                continue
                        else:
                            if confidence < self.config.CONFIDENCE_THRESHOLD:
                                continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.config.VEHICLE_CLASSES[class_id]
                        })
        
        return vehicles
    
    def detect_helmets(self, frame, vehicle_bbox):
        """Detect helmets in a vehicle region"""
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return False
        
        # Expand ROI to better detect helmets
        h, w = vehicle_roi.shape[:2]
        expanded_roi = vehicle_roi[max(0, y1-20):min(frame.shape[0], y2+20), 
                                  max(0, x1-20):min(frame.shape[1], x2+20)]
        
        if expanded_roi.size == 0:
            return False
            
        # Use lower confidence for helmet detection
        results = self.helmet_model(expanded_roi, conf=0.2)
        
        helmet_detected = False
        person_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0].cpu().numpy()
                    
                    if class_id == 0:  # person detected
                        person_detected = True
                    elif class_id == 1:  # helmet detected
                        helmet_detected = True
                    elif class_id == 2:  # no_helmet detected
                        return False
        
        # If person is detected but no helmet, it's a violation
        return person_detected and not helmet_detected
    
    def detect_traffic_light(self, frame):
        """Simple traffic light detection based on color"""
        # Only check top portion of frame where traffic lights typically are
        height, width = frame.shape[:2]
        top_region = frame[0:int(height*0.3), :]
        
        hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        
        # Define more restrictive color ranges for traffic lights
        red_lower = np.array([0, 150, 150])
        red_upper = np.array([10, 255, 255])
        
        yellow_lower = np.array([20, 150, 150])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 150, 150])
        green_upper = np.array([80, 255, 255])
        
        # Check for red light
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_count = cv2.countNonZero(red_mask)
        
        # Check for yellow light
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_count = cv2.countNonZero(yellow_mask)
        
        # Check for green light
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_count = cv2.countNonZero(green_mask)
        
        # More restrictive thresholds
        if red_count > 500:  # Increased threshold
            self.traffic_light_state = 'red'
        elif yellow_count > 500:
            self.traffic_light_state = 'yellow'
        elif green_count > 500:
            self.traffic_light_state = 'green'
        else:
            # Default to green if no clear traffic light detected
            self.traffic_light_state = 'green'
        
        return self.traffic_light_state
    
    def calculate_speed(self, vehicle_id, current_bbox, frame_number):
        """Calculate vehicle speed based on position change"""
        # Use per-track stored positions for more robust speed calculation
        if vehicle_id not in self.vehicle_tracks:
            return 0

        track = self.vehicle_tracks[vehicle_id]

        # Ensure positions deque exists
        if 'positions' not in track:
            track['positions'] = deque(maxlen=30)
            track['frame_numbers'] = deque(maxlen=30)

        # Prefer smoothed centroid stored in the track if available
        if 'centroid' in track and track.get('centroid') is not None:
            center_x, center_y = track['centroid']
        else:
            center_x = (current_bbox[0] + current_bbox[2]) // 2
            center_y = (current_bbox[1] + current_bbox[3]) // 2

        # Append only if position changed meaningfully (avoid duplicates when detection noisy)
        if not track['positions'] or np.hypot(center_x - track['positions'][-1][0], center_y - track['positions'][-1][1]) > 2:
            track['positions'].append((center_x, center_y))
            track['frame_numbers'].append(frame_number)

        # Need at least 3 positions to estimate speed
        if len(track['positions']) > 5:
            prev_pos = track['positions'][-5]
            prev_frame = track['frame_numbers'][-5]
        else:
            prev_pos = track['positions'][0]
            prev_frame = track['frame_numbers'][0]
            curr_pos = track['positions'][-1]
            curr_frame = track['frame_numbers'][-1]

            distance_pixels = np.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            frame_diff = curr_frame - prev_frame
            if frame_diff <= 0:
                return 0

            time_diff = frame_diff / self.config.FPS

            if time_diff > 0 and distance_pixels > 10:  # Minimum movement threshold
                distance_meters = distance_pixels / max(1.0, self.config.PIXELS_PER_METER)
                speed_ms = distance_meters / time_diff
                speed_kmh = speed_ms * 3.6

                # Clamp to reasonable range
                if 0 < speed_kmh < 200:
                    return speed_kmh

        return 0
    
    
    def track_vehicles(self, vehicles, frame=None):
        """Tracking bằng ByteTrack để giữ ID ổn định
        'frame' is optional and only used by some tracker implementations; default is None.
        """

        # Chuyển YOLO detections thành dạng supervision Detections
        xyxy = []
        confidences = []
        class_ids = []

        for v in vehicles:
           x1, y1, x2, y2 = v['bbox']
           xyxy.append([x1, y1, x2, y2])
           confidences.append(v['confidence'])
           class_ids.append(list(self.config.VEHICLE_CLASSES.keys())[list(self.config.VEHICLE_CLASSES.values()).index(v['class_name'])])

        if len(xyxy) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
        )

        # Cập nhật tracker using ByteTrack if available; otherwise fall back to centroid tracker
        if self.tracker is not None and hasattr(self.tracker, 'update_with_detections'):
            try:
                tracks = self.tracker.update_with_detections(detections)

                # Trả về dictionary ID giống như hệ thống cũ
                current_tracks = {}
                for i in range(len(tracks.xyxy)):
                    x1, y1, x2, y2 = map(int, tracks.xyxy[i])
                    track_id = int(tracks.tracker_id[i])
                    class_name = self.config.VEHICLE_CLASSES.get(int(tracks.class_id[i]), "unknown")
                    current_tracks[track_id] = {
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'confidence': float(tracks.confidence[i])
                    }

                return current_tracks
            except Exception as e:
                print(f"Warning: ByteTrack update failed, falling back to centroid tracker: {e}")
                # continue to centroid-based tracker below
                pass

        """Centroid-based tracker with persistence to reduce ID switching
        and centroid smoothing using a short history to reduce jitter."""
        # Build list of detections with centroids
        detections = []  # (bbox, centroid, raw_vehicle)
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((vehicle['bbox'], (cx, cy), vehicle))

        # Use Hungarian (optimal) assignment between predicted track centroids and detections
        matched_tracks = set()
        unmatched_detections = []

        if len(self.vehicle_tracks) > 0 and len(detections) > 0:
            track_ids = list(self.vehicle_tracks.keys())

            # Build predicted centroids for each track using simple linear extrapolation from centroid history
            predicted_centroids = []
            for tid in track_ids:
                track = self.vehicle_tracks[tid]
                hist = track.get('centroid_history')
                if hist and len(hist) >= 2:
                    last = hist[-1]
                    prev = hist[-2]
                    # velocity = last - prev, predict one-step ahead
                    pred = (int(last[0] + (last[0] - prev[0])), int(last[1] + (last[1] - prev[1])))
                else:
                    # fallback to current centroid or center of bbox
                    if 'centroid' in track and track['centroid'] is not None:
                        pred = track['centroid']
                    else:
                        bx = track.get('bbox') or [0, 0, 0, 0]
                        pred = (int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2))
                predicted_centroids.append(pred)

            det_centroids = [det[1] for det in detections]

            # Cost matrix: combine Euclidean distance and (1 - IoU) to prefer overlaps when available.
            # Normalize distance by image diagonal to keep scale comparable to IoU term.
            height, width = 1080, 1920
            diag = math.hypot(width, height)
            cost = np.zeros((len(predicted_centroids), len(det_centroids)), dtype=float)
            for i, p in enumerate(predicted_centroids):
                for j, d in enumerate(det_centroids):
                    dist = np.hypot(p[0] - d[0], p[1] - d[1])
                    # IoU between predicted bbox (use track bbox) and detection bbox if available
                    det_bbox = detections[j][0]
                    track_bbox = self.vehicle_tracks[track_ids[i]].get('bbox') if track_ids[i] in self.vehicle_tracks else None
                    iou_term = 1.0
                    if track_bbox is not None and det_bbox is not None:
                        iou_term = 1.0 - self.calculate_iou(track_bbox, det_bbox)
                    # Combine normalized distance and IoU-term
                    cost[i, j] = (dist / max(1.0, diag)) + iou_term

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            assigned_detections = set()
            assigned_tracks = set()

            for r, c in zip(row_ind, col_ind):
                # convert combined cost back to a gating decision using distance alone
                track_id = track_ids[r]
                det_bbox, det_centroid, raw_vehicle = detections[c]
                # compute actual euclidean distance for gating
                p = predicted_centroids[r]
                d = det_centroids[c]
                dist_px = np.hypot(p[0] - d[0], p[1] - d[1])
                if dist_px <= self.config.MAX_TRACK_DISTANCE:
                    # Update the matched track
                    track = self.vehicle_tracks[track_id]
                    track['bbox'] = det_bbox
                    if 'centroid_history' not in track:
                        cs_window = getattr(self.config, 'CENTROID_SMOOTHING_WINDOW', 5)
                        track['centroid_history'] = deque(maxlen=cs_window)
                    track['centroid_history'].append(det_centroid)
                    hist = track['centroid_history']
                    smoothed = (int(sum([c0[0] for c0 in hist]) / len(hist)), int(sum([c0[1] for c0 in hist]) / len(hist)))
                    track['centroid'] = smoothed
                    track['missing'] = 0
                    track['class_name'] = raw_vehicle.get('class_name')
                    track['confidence'] = raw_vehicle.get('confidence')
                    track['raw'] = raw_vehicle

                    matched_tracks.add(track_id)
                    assigned_detections.add(c)
                    assigned_tracks.add(track_id)
                else:
                    # Do not mark detection as assigned if it's too far; it remains unmatched
                    pass

            # Any detection indices not assigned are unmatched
            for idx, det in enumerate(detections):
                if idx not in assigned_detections:
                    unmatched_detections.append(det)

        else:
            # No existing tracks: all detections are unmatched (create new tracks)
            unmatched_detections = detections[:]

        # Create new tracks for unmatched detections
        for det_bbox, det_centroid, raw_vehicle in unmatched_detections:
            self.track_id_counter += 1
            cs_window = getattr(self.config, 'CENTROID_SMOOTHING_WINDOW', 5)
            self.vehicle_tracks[self.track_id_counter] = {
                'bbox': det_bbox,
                'centroid': det_centroid,
                'centroid_history': deque([det_centroid], maxlen=cs_window),
                'missing': 0,
                'positions': deque(maxlen=30),
                'frame_numbers': deque(maxlen=30),
                'class_name': raw_vehicle.get('class_name'),
                'confidence': raw_vehicle.get('confidence'),
                'raw': raw_vehicle
            }

        # Increase missing frame count for unmatched tracks
        tracks_to_delete = []
        for track_id, track in list(self.vehicle_tracks.items()):
            if track_id not in matched_tracks:
                track['missing'] = track.get('missing', 0) + 1
            # Remove tracks that have been missing for too long
            if track['missing'] > self.config.MAX_MISSING_FRAMES:
                tracks_to_delete.append(track_id)

        for tid in tracks_to_delete:
            del self.vehicle_tracks[tid]

        # Build return structure similar to previous interface: track_id -> raw vehicle info plus bbox
        current_tracks = {}
        for track_id, track in self.vehicle_tracks.items():
            # Ensure bbox is ints
            bbox = [int(v) for v in track['bbox']]
            current_tracks[track_id] = {
                'bbox': bbox,
                'class_name': track.get('class_name', 'unknown'),
                'confidence': track.get('confidence', 0.0)
            }

        return current_tracks
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def check_violations(self, tracked_vehicles, frame_number, frame=None):
        """Check for various traffic violations"""
        violations = []
        
        for track_id, vehicle in tracked_vehicles.items():
            bbox = vehicle['bbox']
            class_name = vehicle['class_name']
            
            # Check red light violation (only for vehicles in motion)
            if self.traffic_light_state == 'red':
                # Only check if vehicle is moving (not stationary)
                speed = self.calculate_speed(track_id, bbox, frame_number)
                if speed > 5:  # Only moving vehicles can violate red light
                    violations.append({
                        'type': 'RED_LIGHT',
                        'description': self.config.VIOLATION_TYPES['RED_LIGHT'],
                        'track_id': track_id,
                        'bbox': bbox,
                        'frame': frame_number,
                        'timestamp': time.time(),
                        'speed': speed
                    })
            
            # Check helmet violation for motorcycles
            if class_name == 'motorcycle' and frame is not None:
                try:
                    no_helmet = self.detect_helmets(frame, bbox)
                    if no_helmet:
                        violations.append({
                            'type': 'NO_HELMET',
                            'description': self.config.VIOLATION_TYPES['NO_HELMET'],
                            'track_id': track_id,
                            'bbox': bbox,
                            'frame': frame_number,
                            'timestamp': time.time()
                        })
                except Exception as e:
                    # Skip helmet detection if there's an error
                    print(f"Helmet detection error: {e}")
                    pass
            
            # Check speeding (only for vehicles with reliable speed data)
            speed = self.calculate_speed(track_id, bbox, frame_number)
            # Initialize per-track speed alert state
            track_state = self.vehicle_tracks.get(track_id)
            if track_state is not None:
                if 'speed_alert_counter' not in track_state:
                    track_state['speed_alert_counter'] = 0
                if 'speed_alerting' not in track_state:
                    track_state['speed_alerting'] = False

                # Hysteresis thresholds
                upper_thresh = self.config.SPEED_LIMIT_KMH
                lower_thresh = max(0, self.config.SPEED_LIMIT_KMH - self.config.SPEED_ALERT_HYSTERESIS_KMH)

                # If currently not alerting and speed above upper thresh, increment counter
                if not track_state['speed_alerting'] and speed > upper_thresh and speed < 200:
                    track_state['speed_alert_counter'] += 1
                    # If sustained high speed over several measurements, trigger alert
                    if track_state['speed_alert_counter'] >= self.config.SPEED_ALERT_CONSISTENT_COUNT:
                        track_state['speed_alerting'] = True
                        violations.append({
                            'type': 'SPEEDING',
                            'description': f"{self.config.VIOLATION_TYPES['SPEEDING']} ({speed:.1f} km/h)",
                            'track_id': track_id,
                            'bbox': bbox,
                            'frame': frame_number,
                            'timestamp': time.time(),
                            'speed': speed
                        })
                else:
                    # If speed drops below lower threshold, reset alerting and counter
                    if speed < lower_thresh:
                        track_state['speed_alert_counter'] = 0
                        track_state['speed_alerting'] = False
                    else:
                        # If still below upper_thresh but above lower, slowly decay counter
                        track_state['speed_alert_counter'] = max(0, track_state.get('speed_alert_counter', 0) - 1)
        
        return violations
    
    def process_frame(self, frame, frame_number):
        """Process a single frame for traffic violations"""
        # Detect traffic light state
        self.detect_traffic_light(frame)
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Track vehicles
        tracked_vehicles = self.track_vehicles(vehicles, frame)
        
        # Check for violations
        violations = self.check_violations(tracked_vehicles, frame_number, frame)
        
        # Add to violation history
        for violation in violations:
            self.violation_history.append(violation)
        
        return tracked_vehicles, violations
    
    def draw_annotations(self, frame, tracked_vehicles, violations):
        """Draw annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw vehicle bounding boxes
        for track_id, vehicle in tracked_vehicles.items():
            bbox = vehicle['bbox']
            class_name = vehicle['class_name']
            confidence = vehicle['confidence']
            
            # Check if this vehicle has violations
            has_violation = any(v['track_id'] == track_id for v in violations)
            color = self.config.VIOLATION_BOX_COLOR if has_violation else self.config.NORMAL_BOX_COLOR
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, self.config.BOX_THICKNESS)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, color, 2)
        
        # Draw violation information
        y_offset = 30
        for violation in violations:
            violation_text = f"VIOLATION: {violation['description']}"
            cv2.putText(annotated_frame, violation_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.VIOLATION_BOX_COLOR, 2)
            y_offset += 30
        
        # Draw traffic light status
        traffic_text = f"Traffic Light: {self.traffic_light_state.upper()}"
        cv2.putText(annotated_frame, traffic_text, (10, annotated_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.TRAFFIC_LIGHT_COLORS.get(self.traffic_light_state, (255, 255, 255)), 2)
        
        return annotated_frame

