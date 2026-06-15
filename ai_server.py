import os
import tempfile
import io
import shutil
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Tối ưu hóa RAM và CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_model = None

# ==========================================
# 1. KỸ THUẬT LAZY LOAD MODEL
# ==========================================
def get_model():
    global ai_model
    if ai_model is None:
        print("⏳ Đang nạp TensorFlow và Mô hình vào RAM...")
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'denoise_softmask_best.h5')
        try:
            ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
            ai_model.predict(dummy_input, verbose=0)
            print("✅ Đã nạp mô hình thành công!")
        except Exception as e:
            print(f"❌ Lỗi tải mô hình: {e}")
            raise e
    return ai_model

# ==========================================
# 2. HÀM XỬ LÝ ÂM THANH CỐT LÕI
# ==========================================
def process_audio(input_path, output_path):
    model = get_model()
    
    y, sr = librosa.load(input_path, sr=16000)
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
# 3. API ENDPOINTS (FastAPI)
# ==========================================

# Sửa lỗi 405 Method Not Allowed bằng cách nhận thêm phương thức HEAD
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "AI Audio Denoiser - FastAPI"}

# BỎ CHỮ ASYNC ĐI! FastAPI sẽ tự động chạy hàm này trên một luồng nền độc lập (Background Thread)
@app.post("/api/clean-audio")
def clean_audio_api(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="File rỗng")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ext = os.path.splitext(audio.filename)[1].lower() or ".webm"
            input_temp_path = os.path.join(temp_dir, f'raw_input{ext}')
            wav_temp_path = os.path.join(temp_dir, 'converted_input.wav')
            output_temp_path = os.path.join(temp_dir, 'clean_output.wav')

            # Đọc và lưu file theo kiểu đồng bộ vì đã bỏ chữ async
            with open(input_temp_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            print("🔄 Đang chuyển đổi định dạng sang WAV...")
            audio_segment = AudioSegment.from_file(input_temp_path)
            audio_segment.export(wav_temp_path, format="wav")

            print("🎙️ Đang xử lý AI denoising...")
            process_audio(wav_temp_path, output_temp_path)
            
            with open(output_temp_path, 'rb') as f:
                return_data = io.BytesIO(f.read())

        return_data.seek(0)
        return StreamingResponse(
            return_data,
            media_type='audio/wav',
            headers={"Content-Disposition": "attachment; filename=clean_audio.wav"}
        )

    except Exception as e:
        import traceback
        print(f"❌ Lỗi: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))