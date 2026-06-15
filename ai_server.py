import os
import tempfile
import io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pydub import AudioSegment
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Khởi tạo Server Flask
app = Flask(__name__)
CORS(app)

# --- THÊM ĐOẠN NÀY ĐỂ ÉP CORS CHO MỌI REQUEST KỂ CẢ KHI CÓ LỖI ---
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response
# -----------------------------------------------------------------

# ==========================================
# 1. TẢI MÔ HÌNH VÀO BỘ NHỚ (Chỉ tải 1 lần khi bật server)
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'denoise_softmask_best.h5')

print("⏳ Đang tải mô hình AI. Vui lòng đợi...")
try:
    # compile=False: Bỏ qua việc build metrics (không cần thiết cho inference)
    # Tránh warning "compile_metrics have yet to be built" và tiết kiệm RAM
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Warm up model: chạy 1 lần với input giả để tránh cold-start chậm ở request đầu tiên
    dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    print("✅ Đã tải và khởi động mô hình thành công!")
except Exception as e:
    print(f"❌ LỖI tải mô hình: {e}")
    model = None


# ==========================================
# 2. HÀM XỬ LÝ ÂM THANH CỐT LÕI
# ==========================================
def process_audio(input_path, output_path):
    # Load file âm thanh, resample về 16000 Hz
    y, sr = librosa.load(input_path, sr=16000)
    print(f"  → Đã load audio: {len(y)} samples, sr={sr}")

    # Bước 1: STFT
    # n_fft=254 → n_fft//2 + 1 = 128 frequency bins (khớp với input model)
    stft = librosa.stft(y, n_fft=254, hop_length=128)
    magnitude = np.abs(stft)
    phase = np.exp(1.j * np.angle(stft))

    freq_bins = magnitude.shape[0]  # Lấy động thay vì hardcode 128
    print(f"  → STFT shape: {magnitude.shape} (freq_bins={freq_bins})")

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
        chunk = norm_spectrogram[:, start: start + window_size]
        chunks.append(chunk)
        start_indices.append(start)

    X_input = np.array(chunks)[..., np.newaxis]
    print(f"  → Số chunks: {num_chunks}, X_input shape: {X_input.shape}")

    # AI Inference
    X_cleaned = model.predict(X_input, verbose=0)
    X_cleaned = np.squeeze(X_cleaned, axis=-1)
    print(f"  → X_cleaned shape: {X_cleaned.shape}")

    # Bước 3: Reconstruction (Dùng freq_bins động thay vì hardcode 128)
    cleaned_padded = np.zeros((freq_bins, padded_time_frames))
    overlap_count = np.zeros((freq_bins, padded_time_frames))

    for i in range(num_chunks):
        start = start_indices[i]
        cleaned_padded[:, start: start + window_size] += X_cleaned[i]
        overlap_count[:, start: start + window_size] += 1

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

    sf.write(output_path, y_clean, sr)
    print(f"  → Đã ghi file output: {output_path}")


# ==========================================
# 3. HEALTH CHECK ENDPOINT (Render cần để xác nhận server đang sống)
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({
        "status": status,
        "service": "AI Audio Denoiser",
        "model_loaded": model is not None
    }), 200


# ==========================================
# 4. API ENDPOINT LỌC ÂM THANH
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
        with tempfile.TemporaryDirectory() as temp_dir:
            # Lấy extension gốc của file để xử lý đúng
            original_filename = audio_file.filename or "raw_input.webm"
            ext = os.path.splitext(original_filename)[1].lower() or ".webm"

            input_temp_path = os.path.join(temp_dir, f'raw_input{ext}')
            wav_temp_path = os.path.join(temp_dir, 'converted_input.wav')
            output_temp_path = os.path.join(temp_dir, 'clean_output.wav')

            # Lưu file gốc
            audio_file.save(input_temp_path)
            print(f"🔄 Đã nhận file: {original_filename} ({os.path.getsize(input_temp_path)} bytes)")

            # Chuyển đổi sang WAV bằng pydub (dùng ffmpeg bên dưới)
            print("🔄 Đang chuyển đổi định dạng sang WAV...")
            audio_segment = AudioSegment.from_file(input_temp_path)
            audio_segment.export(wav_temp_path, format="wav")
            print(f"  → Converted WAV: {os.path.getsize(wav_temp_path)} bytes, duration={len(audio_segment)/1000:.1f}s")

            print("🎙️ Đang xử lý AI denoising...")
            process_audio(wav_temp_path, output_temp_path)
            print("✨ Xử lý xong! Đang đọc vào RAM...")

            with open(output_temp_path, 'rb') as f:
                return_data = io.BytesIO(f.read())

        print("🚀 Đang gửi file sạch về client...")
        return_data.seek(0)

        return send_file(
            return_data,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='clean_audio.wav'
        )

    except Exception as e:
        import traceback
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Chạy server (chỉ dùng khi chạy local, Render dùng Gunicorn)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Server AI đang chạy tại cổng: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)