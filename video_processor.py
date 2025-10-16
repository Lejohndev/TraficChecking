import cv2
import numpy as np
from traffic_detector import TrafficViolationDetector
from config import Config
import os
import json
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        self.config = Config()
        self.detector = TrafficViolationDetector()
        self.violation_log = []
        
    def process_video(self, input_path, output_path=None):
        """Process video file for traffic violations"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        if output_path is None:
            output_path = self.config.OUTPUT_VIDEO_PATH
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure detector uses correct FPS for speed calculations
        try:
            self.detector.config.FPS = fps
        except Exception:
            pass
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        start_time = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracked_vehicles, violations = self.detector.process_frame(frame, frame_number)
                
                # Draw annotations
                annotated_frame = self.detector.draw_annotations(frame, tracked_vehicles, violations)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Log violations
                for violation in violations:
                    self.violation_log.append({
                        'frame': frame_number,
                        'timestamp': datetime.now().isoformat(),
                        'violation': violation
                    })
                
                # Progress update
                frame_number += 1
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
                
                # Optional: Display frame (for debugging)
                # cv2.imshow('Traffic Violation Detection', annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Generate summary report
            self.generate_report()
            
            processing_time = datetime.now() - start_time
            print(f"\nProcessing completed in {processing_time}")
            print(f"Output video saved to: {output_path}")
            print(f"Total violations detected: {len(self.violation_log)}")
    
    def generate_report(self):
        """Generate violation report"""
        if not self.violation_log:
            print("No violations detected")
            return
        
        # Count violations by type
        violation_counts = {}
        for log_entry in self.violation_log:
            violation_type = log_entry['violation']['type']
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        print("\n" + "="*50)
        print("TRAFFIC VIOLATION REPORT")
        print("="*50)
        print(f"Total violations: {len(self.violation_log)}")
        print("\nViolation breakdown:")
        
        for violation_type, count in violation_counts.items():
            description = self.config.VIOLATION_TYPES.get(violation_type, violation_type)
            print(f"  {description}: {count}")
        
        # Save detailed report to JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_violations': len(self.violation_log),
            'violation_counts': violation_counts,
            'violations': self.violation_log
        }
        
        report_file = f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed report saved to: {report_file}")
    
    def process_realtime(self, camera_index=0):
        """Process real-time video from camera"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        print("Starting real-time traffic violation detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracked_vehicles, violations = self.detector.process_frame(frame, frame_number)
                
                # Draw annotations
                annotated_frame = self.detector.draw_annotations(frame, tracked_vehicles, violations)
                
                # Display frame
                cv2.imshow('Real-time Traffic Violation Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_name, annotated_frame)
                    print(f"Screenshot saved: {screenshot_name}")
                
                frame_number += 1
                
        except KeyboardInterrupt:
            print("\nReal-time processing stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def batch_process(self, input_folder, output_folder=None):
        """Process multiple videos in a folder"""
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        if output_folder is None:
            output_folder = os.path.join(input_folder, "processed")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
        
        if not video_files:
            print(f"No video files found in {input_folder}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        
        for i, video_file in enumerate(video_files, 1):
            input_path = os.path.join(input_folder, video_file)
            output_filename = f"processed_{video_file}"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"\nProcessing {i}/{len(video_files)}: {video_file}")
            
            try:
                self.process_video(input_path, output_path)
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                continue
        
        print(f"\nBatch processing completed. Output saved to: {output_folder}")

