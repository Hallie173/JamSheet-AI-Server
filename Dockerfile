# 1. Sử dụng hệ điều hành Linux có cài sẵn Python 3.10 (bản thu gọn)
FROM python:3.10-slim

# 2. Cập nhật hệ thống và cài đặt phần mềm FFmpeg (Bắt buộc để librosa đọc webm)
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    apt-get clean

# 3. Tạo thư mục làm việc bên trong máy chủ ảo
WORKDIR /app

# 4. Copy file thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code và model AI vào máy chủ
COPY . .

# 6. Ra lệnh khởi động server AI
CMD ["python", "ai_server.py"]