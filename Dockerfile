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

# 7. Dùng Gunicorn (production WSGI server) thay cho Flask dev server
#    - 1 worker để tránh load model AI nhiều lần tốn RAM
#    - timeout 120s vì xử lý AI có thể mất vài giây
CMD ["gunicorn", "--workers=1", "--timeout=120", "--bind=0.0.0.0:10000", "ai_server:app"]