#!/usr/bin/env python3
"""
Examples for Traffic Violation Detection System
Các ví dụ sử dụng hệ thống phát hiện vi phạm giao thông
"""

import os
from video_processor import VideoProcessor
from traffic_detector import TrafficViolationDetector
import cv2
import numpy as np

def example_1_single_video():
    """Ví dụ 1: Xử lý một video file"""
    print("Example 1: Processing single video file")
    print("-" * 40)
    
    processor = VideoProcessor()
    
    # Đảm bảo có video để xử lý
    input_video = "demo_traffic.mp4"
    if not os.path.exists(input_video):
        print(f"Video {input_video} not found. Creating demo video...")
        os.system("python run_demo.py")
    
    if os.path.exists(input_video):
        print(f"Processing video: {input_video}")
        processor.process_video(input_video, "output_example1.mp4")
        print("✓ Video processing completed!")
    else:
        print("✗ No video file available for processing")

def example_2_realtime_detection():
    """Ví dụ 2: Phát hiện real-time từ camera"""
    print("\nExample 2: Real-time detection from camera")
    print("-" * 40)
    
    processor = VideoProcessor()
    
    print("Starting real-time detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    try:
        processor.process_realtime(camera_index=0)
        print("✓ Real-time detection completed!")
    except Exception as e:
        print(f"✗ Real-time detection failed: {e}")

def example_3_custom_detector():
    """Ví dụ 3: Sử dụng detector trực tiếp"""
    print("\nExample 3: Using detector directly")
    print("-" * 40)
    
    detector = TrafficViolationDetector()
    
    # Tạo frame test
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Vẽ một chiếc xe giả
    cv2.rectangle(frame, (100, 200), (300, 350), (0, 255, 0), -1)
    cv2.putText(frame, "Test Car", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print("Processing test frame...")
    
    # Phát hiện phương tiện
    vehicles = detector.detect_vehicles(frame)
    print(f"Found {len(vehicles)} vehicles")
    
    # Track phương tiện
    tracked_vehicles = detector.track_vehicles(vehicles)
    print(f"Tracking {len(tracked_vehicles)} vehicles")
    
    # Kiểm tra vi phạm
    violations = detector.check_violations(tracked_vehicles, frame_number=0)
    print(f"Found {len(violations)} violations")
    
    # Vẽ annotations
    annotated_frame = detector.draw_annotations(frame, tracked_vehicles, violations)
    
    # Lưu kết quả
    cv2.imwrite("example3_result.jpg", annotated_frame)
    print("✓ Custom detector example completed! Result saved to example3_result.jpg")

def example_4_batch_processing():
    """Ví dụ 4: Xử lý nhiều video"""
    print("\nExample 4: Batch processing")
    print("-" * 40)
    
    # Tạo thư mục test với video mẫu
    test_folder = "test_videos"
    os.makedirs(test_folder, exist_ok=True)
    
    # Tạo một vài video demo
    print("Creating test videos...")
    os.system("python run_demo.py")  # Tạo demo video
    
    # Copy video demo vào thư mục test
    if os.path.exists("demo_traffic.mp4"):
        import shutil
        shutil.copy("demo_traffic.mp4", os.path.join(test_folder, "traffic_1.mp4"))
        shutil.copy("demo_traffic.mp4", os.path.join(test_folder, "traffic_2.mp4"))
        
        processor = VideoProcessor()
        print(f"Batch processing videos in {test_folder}...")
        processor.batch_process(test_folder, "batch_output")
        print("✓ Batch processing completed!")
    else:
        print("✗ No demo video available for batch processing")

def example_5_custom_configuration():
    """Ví dụ 5: Cấu hình tùy chỉnh"""
    print("\nExample 5: Custom configuration")
    print("-" * 40)
    
    from config import Config
    
    # Tạo config tùy chỉnh
    config = Config()
    
    # Thay đổi một số thông số
    original_speed_limit = config.SPEED_LIMIT_KMH
    original_confidence = config.CONFIDENCE_THRESHOLD
    
    config.SPEED_LIMIT_KMH = 60  # Tăng giới hạn tốc độ
    config.CONFIDENCE_THRESHOLD = 0.7  # Tăng ngưỡng tin cậy
    
    print(f"Original speed limit: {original_speed_limit} km/h")
    print(f"New speed limit: {config.SPEED_LIMIT_KMH} km/h")
    print(f"Original confidence threshold: {original_confidence}")
    print(f"New confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    
    # Tạo detector với config mới
    detector = TrafficViolationDetector()
    detector.config = config  # Áp dụng config mới
    
    print("✓ Custom configuration applied!")
    
    # Khôi phục config gốc
    config.SPEED_LIMIT_KMH = original_speed_limit
    config.CONFIDENCE_THRESHOLD = original_confidence

def run_all_examples():
    """Chạy tất cả các ví dụ"""
    print("="*60)
    print("TRAFFIC VIOLATION DETECTION SYSTEM - EXAMPLES")
    print("="*60)
    
    examples = [
        ("Single Video Processing", example_1_single_video),
        ("Custom Detector Usage", example_3_custom_detector),
        ("Custom Configuration", example_5_custom_configuration),
        ("Batch Processing", example_4_batch_processing),
    ]
    
    for name, example_func in examples:
        print(f"\n{name}")
        print("=" * len(name))
        try:
            example_func()
        except Exception as e:
            print(f"✗ Example failed: {e}")
        print()
    
    print("="*60)
    print("Examples completed!")
    print("\nTo try real-time detection, run:")
    print("  python examples.py --realtime")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--realtime":
        example_2_realtime_detection()
    else:
        run_all_examples()

if __name__ == "__main__":
    main()

