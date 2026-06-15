import os
import tempfile
import io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pydub import AudioSegment  # Thêm thư viện pydub
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Khởi tạo Server Flask
app = Flask(__name__)
# Cho phép Frontend (React) từ mọi nguồn có thể gọi API này
CORS(app)

# ==========================================
# 1. TẢI MÔ HÌNH VÀO BỘ NHỚ (Chỉ tải 1 lần khi bật server)
# ==========================================
# Chú ý: Đổi tên file .h5 hoặc .keras cho đúng với file bạn đang có trong thư mục model/
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'denoise_softmask_best.h5')

print("⏳ Đang tải mô hình AI. Vui lòng đợi...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Đã tải mô hình thành công!")
except Exception as e:
    print(f"❌ LỖI tải mô hình: {e}")
    model = None


# ==========================================
# 2. HÀM XỬ LÝ ÂM THANH CỐT LÕI (Từ test_sound.py của bạn)
# ==========================================
def process_audio(input_path, output_path):
    # Load file
    y, sr = librosa.load(input_path, sr=16000)
    
    # Bước 1: STFT
    stft = librosa.stft(y, n_fft=254, hop_length=128)
    magnitude = np.abs(stft)
    phase = np.exp(1.j * np.angle(stft)) 
    
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    min_val = np.min(log_spectrogram)
    max_val = np.max(log_spectrogram)
    norm_spectrogram = (log_spectrogram - min_val) / (max_val - min_val + 1e-8)
    
    # Bước 2: Chunking (Overlap-Add)
    time_frames = norm_spectrogram.shape[1]
    window_size = 128
    step_size = 32
    
    if time_frames < window_size:
        num_chunks = 1
        padded_time_frames = window_size
    else:
        num_chunks = int(np.ceil((time_frames - window_size) / step_size)) + 1
        padded_time_frames = (num_chunks - 1) * step_size + window_size

    pad_len = padded_time_frames - time_frames
    if pad_len > 0:
        norm_spectrogram = np.pad(norm_spectrogram, ((0, 0), (0, pad_len)), mode='constant')
        
    chunks = []
    start_indices = []
    
    for i in range(num_chunks):
        start = i * step_size
        chunk = norm_spectrogram[:, start : start + window_size]
        chunks.append(chunk)
        start_indices.append(start)
        
    X_input = np.array(chunks)[..., np.newaxis]
    
    # AI Inference
    X_cleaned = model.predict(X_input, verbose=0)
    X_cleaned = np.squeeze(X_cleaned, axis=-1)
    
    # Bước 3: Reconstruction
    cleaned_padded = np.zeros((128, padded_time_frames))
    overlap_count = np.zeros((128, padded_time_frames))
    
    for i in range(num_chunks):
        start = start_indices[i]
        cleaned_padded[:, start : start + window_size] += X_cleaned[i]
        overlap_count[:, start : start + window_size] += 1
        
    cleaned_padded /= np.maximum(overlap_count, 1)
    cleaned_spectrogram_norm = cleaned_padded[:, :time_frames]
    
    cleaned_log_spectrogram = cleaned_spectrogram_norm * (max_val - min_val + 1e-8) + min_val
    cleaned_magnitude = librosa.db_to_amplitude(cleaned_log_spectrogram)
    cleaned_stft = cleaned_magnitude * phase
    
    y_clean = librosa.istft(cleaned_stft, hop_length=128, length=len(y))

    # Bước 4: Normalization
    max_amplitude = np.max(np.abs(y_clean))
    if max_amplitude > 0:
        y_clean = y_clean * (0.8 / max_amplitude)
    
    # Lưu file kết quả
    sf.write(output_path, y_clean, sr)


# ==========================================
# 3. TẠO API ENDPOINT GIAO TIẾP VỚI REACT
# ==========================================
@app.route('/api/clean-audio', methods=['POST'])
def clean_audio_api():
    if model is None:
        return jsonify({"error": "Mô hình AI chưa được tải"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Không tìm thấy file âm thanh đính kèm"}), 400

    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "File rỗng"}), 400

    try:
        # Tạo thư mục tạm để lưu file người dùng gửi lên và file AI xuất ra
        with tempfile.TemporaryDirectory() as temp_dir:
            input_temp_path = os.path.join(temp_dir, 'raw_input.webm')
            wav_temp_path = os.path.join(temp_dir, 'converted_input.wav')
            output_temp_path = os.path.join(temp_dir, 'clean_output.wav')
            
            # Lưu file gốc (webm)
            audio_file.save(input_temp_path)
            
            # Chuyển đổi webm sang wav bằng pydub trước khi đưa vào librosa
            print("🔄 Đang chuyển đổi định dạng WebM sang WAV...")
            audio_segment = AudioSegment.from_file(input_temp_path)
            audio_segment.export(wav_temp_path, format="wav")
            
            # Gọi hàm xử lý AI với tệp wav đã được chuyển đổi
            print("🎙️ Đã nhận yêu cầu lọc ồn. Đang xử lý...")
            process_audio(wav_temp_path, output_temp_path)
            print("✨ Xử lý xong! Đang nạp dữ liệu vào RAM...")
            
            # Đọc file âm thanh từ ổ cứng vào bộ nhớ RAM (BytesIO)
            with open(output_temp_path, 'rb') as f:
                return_data = io.BytesIO(f.read())
                
        # Ngay tại dòng này (khi thoát khỏi thụt lề của chữ "with"), 
        # Python sẽ tự động xóa sạch file trên ổ cứng.
        # Rất an toàn, nhưng ta đã lưu dữ liệu vào biến `return_data` rồi!

        print("🚀 Đang gửi trả file sạch về cho Web...")
        return_data.seek(0) # Đưa con trỏ chuột về đầu file để Flask đọc
        
        return send_file(
            return_data,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='clean_audio.wav'
        )
                
    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        return jsonify({"error": str(e)}), 500

# Chạy Server
if __name__ == '__main__':
    # Render sẽ cấp một cổng động thông qua biến môi trường PORT, 
    # nếu không có (chạy local) thì mặc định dùng 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Server AI đang chạy tại cổng: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)