# 1. Sử dụng hệ điều hành Linux có cài sẵn Python 3.10 (bản thu gọn)
FROM python:3.10-slim

# 2. Cập nhật hệ thống và cài đặt phần mềm FFmpeg (Bắt buộc để pydub/librosa đọc webm)
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Tạo thư mục làm việc bên trong máy chủ ảo
WORKDIR /app

# 4. Copy file thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code và model AI vào máy chủ
COPY . .

# 6. Expose cổng 10000
EXPOSE 10000

# 7. Dùng Gunicorn với cấu hình tối ưu cho AI workload:
#    - 1 worker: tránh load model nhiều lần, tiết kiệm RAM
#    - worker-class gthread: hỗ trợ xử lý đồng thời tốt hơn sync cho I/O nặng
#    - threads=2: cho phép 2 luồng song song trong cùng 1 worker
#    - timeout=300: tăng lên 5 phút vì CPU inference có thể chậm trên Render free tier
#    - graceful-timeout=30: thời gian để worker hoàn thành request trước khi shutdown
CMD ["gunicorn", "--workers=1", "--worker-class=gthread", "--threads=2", "--timeout=300", "--graceful-timeout=30", "--bind=0.0.0.0:10000", "ai_server:app"]