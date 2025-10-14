# Traffic Violation Detection System

Hệ thống phát hiện vi phạm giao thông sử dụng Computer Vision và YOLO để phân tích video giao thông và phát hiện các vi phạm như chạy đèn đỏ, không đội mũ bảo hiểm, và vượt tốc độ.

## Tính năng chính

- **Phát hiện phương tiện**: Sử dụng YOLOv8 để phát hiện xe ô tô, xe máy, xe buýt, xe tải
- **Phát hiện vi phạm giao thông**:
  - Chạy đèn đỏ
  - Không đội mũ bảo hiểm (xe máy)
  - Vượt tốc độ
  - Đi sai làn đường
- **Tracking phương tiện**: Theo dõi phương tiện qua các frame để tính toán tốc độ
- **Phát hiện đèn giao thông**: Nhận diện trạng thái đèn giao thông (đỏ, vàng, xanh)
- **Xử lý video**: Hỗ trợ xử lý video file, real-time từ camera, và batch processing
- **Báo cáo vi phạm**: Tạo báo cáo chi tiết về các vi phạm được phát hiện

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- OpenCV
- PyTorch
- YOLOv8

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Tải model YOLO

Hệ thống sẽ tự động tải YOLOv8 model lần đầu chạy.

## Sử dụng

### 1. Xử lý video file

```bash
# Xử lý video đơn lẻ
python main.py --mode video --input input_video.mp4 --output output_video.mp4

# Hoặc đơn giản hơn
python main.py --input your_video.mp4
```

### 2. Xử lý real-time từ camera

```bash
# Sử dụng camera mặc định (index 0)
python main.py --mode realtime

# Sử dụng camera khác
python main.py --mode realtime --camera 1
```

### 3. Batch processing (xử lý nhiều video)

```bash
python main.py --mode batch --folder /path/to/video/folder
```

### 4. Demo mode

```bash
# Chạy demo (không cần tham số)
python main.py
```

## Cấu hình

Chỉnh sửa file `config.py` để thay đổi các thông số:

```python
class Config:
    # Ngưỡng tin cậy cho detection
    CONFIDENCE_THRESHOLD = 0.5

    # Giới hạn tốc độ (km/h)
    SPEED_LIMIT_KMH = 50

    # Tỷ lệ pixel/mét (cần hiệu chỉnh theo camera)
    PIXELS_PER_METER = 10
```

## Cấu trúc project

```
TraficChecking/
├── main.py                 # Script chính
├── traffic_detector.py     # Class phát hiện vi phạm giao thông
├── video_processor.py      # Class xử lý video
├── config.py              # File cấu hình
├── requirements.txt       # Dependencies
└── README.md             # Hướng dẫn sử dụng
```

## Các loại vi phạm được phát hiện

### 1. Chạy đèn đỏ

- Phát hiện khi xe di chuyển qua giao lộ khi đèn đỏ
- Sử dụng color detection để nhận diện trạng thái đèn giao thông

### 2. Không đội mũ bảo hiểm

- Phát hiện người điều khiển xe máy không đội mũ bảo hiểm
- Sử dụng YOLO để detect helmet trong vùng xe máy

### 3. Vượt tốc độ

- Tính toán tốc độ dựa trên sự thay đổi vị trí của xe
- So sánh với giới hạn tốc độ được cấu hình

## Output

### Video output

- Video gốc với các annotation:
  - Bounding box cho phương tiện (xanh = bình thường, đỏ = vi phạm)
  - ID tracking cho mỗi phương tiện
  - Thông tin vi phạm trên màn hình
  - Trạng thái đèn giao thông

### Báo cáo vi phạm

- File JSON chứa chi tiết các vi phạm:
  - Thời gian vi phạm
  - Loại vi phạm
  - ID phương tiện
  - Tọa độ bounding box
  - Tốc độ (nếu là vi phạm tốc độ)

## Hiệu chỉnh hệ thống

### 1. Hiệu chỉnh tốc độ

Thay đổi `PIXELS_PER_METER` trong `config.py` để hiệu chỉnh tính toán tốc độ:

```python
PIXELS_PER_METER = 10  # Tăng/giảm theo camera và góc quay
```

### 2. Hiệu chỉnh phát hiện đèn giao thông

Có thể cần điều chỉnh các threshold màu trong hàm `detect_traffic_light()`:

```python
# Trong traffic_detector.py
red_count > 100  # Tăng/giảm threshold
```

### 3. Cải thiện độ chính xác

- Tăng `CONFIDENCE_THRESHOLD` để giảm false positive
- Giảm `CONFIDENCE_THRESHOLD` để tăng detection rate
- Điều chỉnh `IOU_THRESHOLD` cho tracking

## Troubleshooting

### Lỗi thường gặp

1. **"CUDA out of memory"**

   - Giảm kích thước video hoặc sử dụng CPU
   - Thêm `device='cpu'` trong YOLO model

2. **Video không mở được**

   - Kiểm tra format video (hỗ trợ: mp4, avi, mov, mkv)
   - Cài đặt codec cần thiết

3. **Camera không hoạt động**

   - Kiểm tra camera index (0, 1, 2...)
   - Đảm bảo camera không bị sử dụng bởi ứng dụng khác

4. **Model không tải được**
   - Kiểm tra kết nối internet
   - Model sẽ được tải tự động lần đầu

## Mở rộng

### Thêm loại vi phạm mới

1. Thêm logic detection trong `check_violations()`
2. Cập nhật `VIOLATION_TYPES` trong `config.py`
3. Thêm visualization trong `draw_annotations()`

### Sử dụng model khác

- Thay đổi `YOLO_MODEL` trong `config.py`
- Có thể sử dụng YOLOv8s, YOLOv8m, YOLOv8l cho độ chính xác cao hơn

## License

MIT License

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub.
