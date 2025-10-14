#!/usr/bin/env python3
"""
Demo script for Traffic Violation Detection System
Tạo video demo với các vi phạm giao thông giả lập
"""

import cv2
import numpy as np
import os

def create_demo_video():
    """Tạo video demo với các vi phạm giao thông giả lập"""
    
    # Video settings
    width, height = 1280, 720
    fps = 30
    duration = 30  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_traffic.mp4', fourcc, fps, (width, height))
    
    print("Creating demo traffic video...")
    
    for frame_num in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (0, height//2 - 50), (width, height//2 + 50), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, height//2 - 5), (width, height//2 + 5), (255, 255, 255), -1)
        
        # Draw lane markings
        for i in range(0, width, 100):
            cv2.rectangle(frame, (i, height//2 - 2), (i + 50, height//2 + 2), (255, 255, 255), -1)
        
        # Draw traffic light
        light_x, light_y = 100, 100
        
        # Traffic light states (changes every 5 seconds)
        light_state = (frame_num // (fps * 5)) % 3
        
        # Traffic light pole
        cv2.rectangle(frame, (light_x - 10, light_y), (light_x + 10, light_y + 200), (50, 50, 50), -1)
        
        # Traffic light box
        cv2.rectangle(frame, (light_x - 20, light_y), (light_x + 20, light_y + 60), (80, 80, 80), -1)
        cv2.rectangle(frame, (light_x - 20, light_y), (light_x + 20, light_y + 60), (255, 255, 255), 2)
        
        # Traffic light colors
        if light_state == 0:  # Red
            cv2.circle(frame, (light_x, light_y + 20), 8, (0, 0, 255), -1)
            cv2.circle(frame, (light_x, light_y + 40), 8, (40, 40, 40), -1)
        elif light_state == 1:  # Yellow
            cv2.circle(frame, (light_x, light_y + 20), 8, (40, 40, 40), -1)
            cv2.circle(frame, (light_x, light_y + 40), 8, (0, 255, 255), -1)
        else:  # Green
            cv2.circle(frame, (light_x, light_y + 20), 8, (40, 40, 40), -1)
            cv2.circle(frame, (light_x, light_y + 40), 8, (0, 255, 0), -1)
        
        # Draw vehicles
        # Car 1 - moves across screen
        car1_x = int((frame_num * 5) % (width + 100)) - 50
        car1_y = height//2 - 30
        
        if 0 <= car1_x <= width:
            # Car body
            cv2.rectangle(frame, (car1_x, car1_y), (car1_x + 80, car1_y + 40), (0, 0, 255), -1)
            # Car windows
            cv2.rectangle(frame, (car1_x + 10, car1_y + 5), (car1_x + 70, car1_y + 25), (200, 200, 255), -1)
            # Wheels
            cv2.circle(frame, (car1_x + 15, car1_y + 35), 8, (50, 50, 50), -1)
            cv2.circle(frame, (car1_x + 65, car1_y + 35), 8, (50, 50, 50), -1)
            
            # Add label
            cv2.putText(frame, "Car 1", (car1_x, car1_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Motorcycle - moves slower
        bike_x = int((frame_num * 3) % (width + 100)) - 50
        bike_y = height//2 + 10
        
        if 0 <= bike_x <= width:
            # Bike body
            cv2.rectangle(frame, (bike_x, bike_y), (bike_x + 50, bike_y + 30), (0, 100, 200), -1)
            # Wheels
            cv2.circle(frame, (bike_x + 10, bike_y + 25), 6, (50, 50, 50), -1)
            cv2.circle(frame, (bike_x + 40, bike_y + 25), 6, (50, 50, 50), -1)
            
            # Add label
            cv2.putText(frame, "Bike", (bike_x, bike_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add violation text when appropriate
        if light_state == 0 and car1_x > 200:  # Red light violation
            cv2.putText(frame, "VIOLATION: Red Light Running", (400, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}", (width - 150, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add traffic light status
        status_text = ["RED", "YELLOW", "GREEN"][light_state]
        cv2.putText(frame, f"Traffic Light: {status_text}", (width - 250, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print("Demo video created: demo_traffic.mp4")

if __name__ == "__main__":
    create_demo_video()

