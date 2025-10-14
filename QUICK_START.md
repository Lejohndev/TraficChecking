# Hướng dẫn nhanh - Traffic Violation Detection

## 🚀 Cài đặt nhanh

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Tạo video demo
python run_demo.py

# 3. Chạy hệ thống
python main.py
```

## 📹 Sử dụng cơ bản

### Xử lý video file

```bash
python main.py --input your_video.mp4
```

### Real-time từ camera

```bash
python main.py --mode realtime
```

### Batch processing

```bash
python main.py --mode batch --folder /path/to/videos
```

## 🔧 Test hệ thống

```bash
python test_system.py
```

## 📚 Xem ví dụ

```bash
python examples.py
```

## ⚡ Tính năng chính

- ✅ Phát hiện xe ô tô, xe máy, xe buýt, xe tải
- ✅ Phát hiện chạy đèn đỏ
- ✅ Phát hiện không đội mũ bảo hiểm
- ✅ Phát hiện vượt tốc độ
- ✅ Tracking phương tiện qua các frame
- ✅ Tạo báo cáo vi phạm chi tiết
- ✅ Xử lý real-time từ camera
- ✅ Batch processing nhiều video

## 📁 Cấu trúc file

```
TraficChecking/
├── main.py              # Script chính
├── traffic_detector.py  # Class phát hiện vi phạm
├── video_processor.py   # Class xử lý video
├── config.py           # Cấu hình hệ thống
├── run_demo.py         # Tạo video demo
├── test_system.py      # Test hệ thống
├── examples.py         # Ví dụ sử dụng
├── install.bat         # Script cài đặt Windows
└── README.md           # Hướng dẫn chi tiết
```

## 🎯 Output

- **Video**: Video gốc với annotations vi phạm
- **Báo cáo**: File JSON chứa chi tiết vi phạm
- **Log**: Thông tin xử lý và thống kê

## ⚠️ Lưu ý

1. **Camera**: Đảm bảo camera không bị sử dụng bởi ứng dụng khác
2. **Video format**: Hỗ trợ mp4, avi, mov, mkv, wmv
3. **Performance**: Sử dụng GPU nếu có để tăng tốc độ
4. **Calibration**: Cần hiệu chỉnh `PIXELS_PER_METER` trong config.py

## 🆘 Troubleshooting

- **Lỗi camera**: Thử đổi camera index (0, 1, 2...)
- **Lỗi video**: Kiểm tra format và codec
- **Lỗi model**: Kiểm tra kết nối internet để tải YOLO
- **Performance chậm**: Giảm kích thước video hoặc sử dụng model nhỏ hơn

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:

1. Chạy `python test_system.py` để kiểm tra
2. Xem `README.md` để biết chi tiết
3. Chạy `python examples.py` để xem ví dụ
