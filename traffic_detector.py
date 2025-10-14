import cv2
import numpy as np
from ultralytics import YOLO
from config import Config
import time
from collections import defaultdict, deque

class TrafficViolationDetector:
    def __init__(self):
        self.config = Config()
        self.vehicle_model = YOLO(self.config.YOLO_MODEL)
        self.helmet_model = YOLO(self.config.HELMET_MODEL)
        
        # Track vehicles across frames
        self.vehicle_tracks = {}
        self.track_id_counter = 0
        
        # Violation tracking
        self.violations = []
        self.violation_history = deque(maxlen=100)
        
        # Traffic light state
        self.traffic_light_state = 'unknown'
        
        # Speed calculation
        self.speed_estimates = {}
        
    def detect_vehicles(self, frame):
        """Detect vehicles in frame using YOLO"""
        results = self.vehicle_model(frame, conf=self.config.CONFIDENCE_THRESHOLD)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for vehicle classes only
                    if int(box.cls[0]) in self.config.VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': int(box.cls[0]),
                            'class_name': self.config.VEHICLE_CLASSES[int(box.cls[0])]
                        })
        
        return vehicles
    
    def detect_helmets(self, frame, vehicle_bbox):
        """Detect helmets in a vehicle region"""
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return False
            
        results = self.helmet_model(vehicle_roi, conf=0.3)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 1:  # helmet detected
                        return True
        
        return False
    
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
        if vehicle_id not in self.speed_estimates:
            self.speed_estimates[vehicle_id] = {
                'positions': deque(maxlen=15),
                'frame_numbers': deque(maxlen=15)
            }
        
        center_x = (current_bbox[0] + current_bbox[2]) // 2
        center_y = (current_bbox[1] + current_bbox[3]) // 2
        
        self.speed_estimates[vehicle_id]['positions'].append((center_x, center_y))
        self.speed_estimates[vehicle_id]['frame_numbers'].append(frame_number)
        
        # Need more frames for accurate speed calculation
        if len(self.speed_estimates[vehicle_id]['positions']) >= 5:
            # Calculate distance moved over last 5 frames
            prev_pos = self.speed_estimates[vehicle_id]['positions'][-5]
            curr_pos = self.speed_estimates[vehicle_id]['positions'][-1]
            
            distance_pixels = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            
            # Calculate time difference (5 frames)
            frame_diff = 5
            time_diff = frame_diff / self.config.FPS
            
            if time_diff > 0 and distance_pixels > 10:  # Minimum movement threshold
                # Convert to km/h (rough estimation)
                distance_meters = distance_pixels / self.config.PIXELS_PER_METER
                speed_ms = distance_meters / time_diff
                speed_kmh = speed_ms * 3.6
                
                # Only return realistic speeds (0-200 km/h)
                if 0 < speed_kmh < 200:
                    return speed_kmh
        
        return 0
    
    def track_vehicles(self, vehicles):
        """Simple vehicle tracking using IoU"""
        if not hasattr(self, 'prev_vehicles'):
            self.prev_vehicles = {}
            self.track_id_counter = 0
        
        current_tracks = {}
        
        for vehicle in vehicles:
            best_match_id = None
            best_iou = 0
            
            for track_id, prev_vehicle in self.prev_vehicles.items():
                iou = self.calculate_iou(vehicle['bbox'], prev_vehicle['bbox'])
                if iou > best_iou and iou > self.config.IOU_THRESHOLD:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                current_tracks[best_match_id] = vehicle
            else:
                self.track_id_counter += 1
                current_tracks[self.track_id_counter] = vehicle
        
        self.prev_vehicles = current_tracks
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
            
            # Check helmet violation for motorcycles (simplified check)
            if class_name == 'motorcycle' and frame is not None:
                # For now, skip helmet detection as it's not reliable
                # has_helmet = self.detect_helmets(frame, bbox)
                # if not has_helmet:
                #     violations.append({...})
                pass
            
            # Check speeding (only for vehicles with reliable speed data)
            speed = self.calculate_speed(track_id, bbox, frame_number)
            if speed > self.config.SPEED_LIMIT_KMH and speed < 200:  # Realistic speed range
                violations.append({
                    'type': 'SPEEDING',
                    'description': f"{self.config.VIOLATION_TYPES['SPEEDING']} ({speed:.1f} km/h)",
                    'track_id': track_id,
                    'bbox': bbox,
                    'frame': frame_number,
                    'timestamp': time.time(),
                    'speed': speed
                })
        
        return violations
    
    def process_frame(self, frame, frame_number):
        """Process a single frame for traffic violations"""
        # Detect traffic light state
        self.detect_traffic_light(frame)
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Track vehicles
        tracked_vehicles = self.track_vehicles(vehicles)
        
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
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
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

