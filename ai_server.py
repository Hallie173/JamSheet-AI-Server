import os
import tempfile
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Tối ưu hóa RAM và CPU cho server cloud miễn phí
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

# Khởi tạo Server Flask
app = Flask(__name__)
CORS(app)

# Ép CORS Header cho mọi response để tránh lỗi Fake CORS của trình duyệt
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Biến toàn cục chứa model
ai_model = None

# ==========================================
# 1. KỸ THUẬT LAZY LOAD MODEL
# ==========================================
def get_model():
    global ai_model
    if ai_model is None:
        print("⏳ LẦN CHẠY ĐẦU TIÊN: Đang nạp TensorFlow và Mô hình vào RAM...")
        # Import tensorflow ở đây để Flask khởi động nhanh chớp nhoáng
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU') # Ép chạy CPU để tránh lỗi Driver
        
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'denoise_softmask_best.h5')
        try:
            ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
            ai_model.predict(dummy_input, verbose=0)
            print("✅ Đã nạp mô hình thành công! Các lần sau sẽ chạy ngay lập tức.")
        except Exception as e:
            print(f"❌ Lỗi tải mô hình: {e}")
            raise e
    return ai_model


# ==========================================
# 2. HÀM XỬ LÝ ÂM THANH CỐT LÕI
# ==========================================
def process_audio(input_path, output_path):
    # Gọi hàm để lấy model (Nếu đã nạp rồi thì sẽ trả về luôn)
    model = get_model()
    
    y, sr = librosa.load(input_path, sr=16000)
    print(f"  → Đã load audio: {len(y)} samples, sr={sr}")

    stft = librosa.stft(y, n_fft=254, hop_length=128)
    magnitude = np.abs(stft)
    phase = np.exp(1.j * np.angle(stft))

    freq_bins = magnitude.shape[0]

    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    min_val = np.min(log_spectrogram)
    max_val = np.max(log_spectrogram)
    norm_spectrogram = (log_spectrogram - min_val) / (max_val - min_val + 1e-8)

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
    
    # Dùng model đã nạp để predict
    X_cleaned = model.predict(X_input, verbose=0)
    X_cleaned = np.squeeze(X_cleaned, axis=-1)

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

    max_amplitude = np.max(np.abs(y_clean))
    if max_amplitude > 0:
        y_clean = y_clean * (0.8 / max_amplitude)

    sf.write(output_path, y_clean, sr)


# ==========================================
# 3. HEALTH CHECK & API ENDPOINT
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    # Ping nhanh để Render biết server đang sống mà không cần chờ load model
    return jsonify({
        "status": "ok",
        "service": "AI Audio Denoiser",
        "model_loaded": ai_model is not None
    }), 200


@app.route('/api/clean-audio', methods=['POST'])
def clean_audio_api():
    if 'audio' not in request.files:
        return jsonify({"error": "Không tìm thấy file âm thanh đính kèm"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "File rỗng"}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_filename = audio_file.filename or "raw_input.webm"
            ext = os.path.splitext(original_filename)[1].lower() or ".webm"

            input_temp_path = os.path.join(temp_dir, f'raw_input{ext}')
            wav_temp_path = os.path.join(temp_dir, 'converted_input.wav')
            output_temp_path = os.path.join(temp_dir, 'clean_output.wav')

            audio_file.save(input_temp_path)
            
            print("🔄 Đang chuyển đổi định dạng sang WAV...")
            audio_segment = AudioSegment.from_file(input_temp_path)
            audio_segment.export(wav_temp_path, format="wav")

            print("🎙️ Đang xử lý AI denoising...")
            process_audio(wav_temp_path, output_temp_path)
            
            with open(output_temp_path, 'rb') as f:
                return_data = io.BytesIO(f.read())

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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Server AI đang chạy tại cổng: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)