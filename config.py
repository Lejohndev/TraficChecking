import os

class Config:
    # Model paths
    YOLO_MODEL = "yolo12s.pt"  # YOLOv8 nano model for vehicle detection
    HELMET_MODEL = "helmet.pt"  # Updated helmet detection model
    # Video settings
    INPUT_VIDEO_PATH = "input_video.mp4"
    OUTPUT_VIDEO_PATH = "output_violations.mp4"
    FPS = 30
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.3 # Lower threshold for better motorcycle detection
    IOU_THRESHOLD = 0.45
    MOTORCYCLE_CONFIDENCE_THRESHOLD = 0.15  # Very low for motorcycles
    
    # Traffic light detection
    TRAFFIC_LIGHT_COLORS = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255), 
        'green': (0, 255, 0)
    }
    
    # Violation types
    VIOLATION_TYPES = {
        'RED_LIGHT': 'Chạy đèn đỏ',
        'SPEEDING': 'Vượt tốc độ',
        'NO_HELMET': 'Không đội mũ bảo hiểm',
        'WRONG_LANE': 'Đi sai làn đường',
        'NO_SEATBELT': 'Không thắt dây an toàn'
    }
    
    # Vehicle classes in COCO dataset
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Motorcycle-specific settings
    MOTORCYCLE_CLASS_ID = 3
    
    # Helmet detection classes
    HELMET_CLASSES = {
        1: 'helmet',
        2: 'no_helmet'
    }
    
    # Speed calculation settings
    SPEED_LIMIT_KMH = 50  # Default speed limit
    PIXELS_PER_METER = 10  # Approximate pixels per meter (needs calibration)
    # Tracking settings
    MAX_TRACK_DISTANCE = 200  # Max pixel distance to associate detections to existing tracks
    MAX_MISSING_FRAMES = 12    # How many frames a track can miss before being removed
    # Require this many consistent detections to confirm a new track (reduces transient IDs)
    TRACK_CONFIRMATION_FRAMES = 3
    # Region of interest (ROI) for detection: either pixels (x1,y1,x2,y2) or relative (0..1) fractions
    # If ROI_RELATIVE is True, ROI is interpreted as fractions of (width, height): (x1, y1, x2, y2)
    ROI_RELATIVE = True
    ROI = (0.0, 0.30, 1.0, 0.95)  # default: ignore top 35% (sky), focus on road area
    # Speed alert tuning
    SPEED_ALERT_CONSISTENT_COUNT = 3  # number of consecutive measurements above limit before alert
    SPEED_ALERT_HYSTERESIS_KMH = 5    # reduce false toggles by requiring drop below (limit - hysteresis)
    # Centroid smoothing
    CENTROID_SMOOTHING_WINDOW = 9  # number of recent centroids to average for smoothing
    
    # Helmet detection settings - Tối ưu hóa để giảm false positive
    HELMET_CONFIDENCE_THRESHOLD = 0.3  # Giảm từ 0.5 xuống 0.3 để tránh miss detection
    HELMET_VIOLATION_CONFIDENCE = 0.4  # Confidence tối thiểu để xác định vi phạm
    HELMET_BUFFER_SIZE = 30  # Số frame để đánh giá (tăng từ 20)
    HELMET_MIN_FRAMES = 25  # Số frame tối thiểu để đánh giá
    HELMET_VIOLATION_RATIO = 0.7  # Tỷ lệ vi phạm tối thiểu (70%)
    HELMET_REQUIRED_VIOLATIONS = 20  # Số frame vi phạm tối thiểu
    
    # Helmet detection thresholds - Điều chỉnh để chính xác hơn
    HELMET_DETECTION_CONFIDENCE = 0.5  # Confidence tối thiểu để phát hiện helmet
    NO_HELMET_DETECTION_CONFIDENCE = 0.7  # Confidence tối thiểu để phát hiện no-helmet
    
    # Visualization settings
    VIOLATION_BOX_COLOR = (0, 0, 255)  # Red for violations
    NORMAL_BOX_COLOR = (0, 255, 0)    # Green for normal vehicles
    TEXT_COLOR = (255, 255, 255)      # White text
    BOX_THICKNESS = 2
    FONT_SCALE = 0.6

