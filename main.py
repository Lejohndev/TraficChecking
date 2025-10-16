#!/usr/bin/env python3
"""
Traffic Violation Detection System
Main script to run the traffic violation detection system
"""

import argparse
import os
import sys
import platform

# On Windows with very new Python versions numpy/OpenCV wheels can be incompatible
# (e.g. numpy built with MINGW). Detect that early and provide actionable guidance.
if sys.version_info >= (3, 13) and os.name == 'nt' and os.environ.get('SKIP_PYTHON_BINARY_IMPORTS', '0') != '1':
    print(f"\nWARNING: Detected Python {platform.python_version()} on Windows.")
    print("Some binary packages (numpy/OpenCV) may be built with MINGW and cause low-level errors.")
    print("Recommended: run this project in Python 3.10 or 3.11 in a venv or conda environment.")
    print("If you understand the risk and want to proceed, set SKIP_PYTHON_BINARY_IMPORTS=1 in environment.")
    sys.exit(1)

from config import Config

def main():
    parser = argparse.ArgumentParser(description='Traffic Violation Detection System')
    parser.add_argument('--mode', choices=['video', 'realtime', 'batch'], 
                       default='video', help='Processing mode')
    parser.add_argument('--input', '-i', help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path')
    parser.add_argument('--camera', '-c', type=int, default=0, 
                       help='Camera index for real-time mode')
    parser.add_argument('--folder', '-f', help='Input folder for batch processing')
    
    args = parser.parse_args()
    
    # Import processor at runtime to surface import errors clearly (avoid NameError)
    try:
        from video_processor import VideoProcessor
    except Exception as e:
        print("Error: failed to import VideoProcessor (likely numpy/OpenCV/YOLO binary issue)")
        print(str(e))
        print("Fix: run this project in Python 3.10/3.11 inside a virtualenv or conda, or set SKIP_PYTHON_BINARY_IMPORTS=1 to bypass the version guard.")
        return 1

    # Initialize processor
    processor = VideoProcessor()
    
    try:
        if args.mode == 'video':
            # Single video processing
            if not args.input:
                print("Error: Input video file required for video mode")
                print("Usage: python main.py --mode video --input input_video.mp4")
                return
            
            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                return
            
            print("Starting video processing...")
            processor.process_video(args.input, args.output)
            
        elif args.mode == 'realtime':
            # Real-time processing from camera
            print("Starting real-time processing...")
            processor.process_realtime(args.camera)
            
        elif args.mode == 'batch':
            # Batch processing
            if not args.folder:
                print("Error: Input folder required for batch mode")
                print("Usage: python main.py --mode batch --folder /path/to/videos")
                return
            
            print("Starting batch processing...")
            processor.batch_process(args.folder, args.output)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

def demo():
    """Demo function with example usage"""
    print("Traffic Violation Detection System Demo")
    print("="*50)
    
    # Check if demo video exists
    demo_video = "demo_traffic.mp4"
    if os.path.exists(demo_video):
        print(f"Processing demo video: {demo_video}")
        try:
            from video_processor import VideoProcessor
        except Exception as e:
            print("Error importing VideoProcessor for demo (likely binary dependency issue):")
            print(str(e))
            print("Use Python 3.10/3.11 in a venv/conda or set SKIP_PYTHON_BINARY_IMPORTS=1 to bypass guard.")
            return
        processor = VideoProcessor()
        processor.process_video(demo_video, "demo_output.mp4")
    else:
        print(f"Demo video '{demo_video}' not found.")
        print("Please place a traffic video file named 'demo_traffic.mp4' in the current directory")
        print("or use your own video with:")
        print("python main.py --mode video --input your_video.mp4")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo()
    else:
        # Run with arguments
        sys.exit(main())

