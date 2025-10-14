@echo off
echo Installing Traffic Violation Detection System...
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Downloading YOLO models...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo.
echo Creating demo video...
python run_demo.py

echo.
echo Installation completed!
echo.
echo To run the system:
echo   python main.py --input your_video.mp4
echo   python main.py --mode realtime
echo   python main.py (for demo)
echo.
pause

