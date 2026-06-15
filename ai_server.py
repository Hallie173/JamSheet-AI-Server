import os
import tempfile
import io
import gc
import shutil
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 1. TỐI ƯU HÓA PHẦN CỨNG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. TẢI MÔ HÌNH TOÀN CỤC VÀO RAM
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'denoise_softmask_best.h5')
try:
    # flush=True ép log hiển thị ngay lập tức
    print("⏳ Đang tải mô hình AI vào RAM...", flush=True)
    ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    ai_model.predict(np.zeros((1, 128, 128, 1), dtype=np.float32), verbose=0)
    print("✅ Mô hình đã sẵn sàng!", flush=True)
except Exception as e:
    print(f"❌ Lỗi tải mô hình: {e}", flush=True)
    ai_model = None

# ==========================================
# 3. HÀM XỬ LÝ ÂM THANH CỐT LÕI
# ==========================================
# Tham số STFT — phải nhất quán giữa forward và inverse transform
N_FFT = 256          # power-of-2 → FFT nhanh; 256//2+1 = 129 freq bins
HOP_LENGTH = 64      # = N_FFT//4 → overlap 75%, giúp istft mượt mà
MODEL_FREQ_BINS = 128  # số freq bins model mong đợi

def process_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)

    # --- STFT (forward) ---
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
    magnitude = np.abs(stft)   # shape: (N_FFT//2+1, T) = (129, T)
    phase = np.exp(1.j * np.angle(stft))

    # Cắt freq dimension về MODEL_FREQ_BINS để khớp với input model
    magnitude_model = magnitude[:MODEL_FREQ_BINS, :]   # (128, T)
    phase_model     = phase[:MODEL_FREQ_BINS, :]       # (128, T)
    freq_bins = MODEL_FREQ_BINS

    # --- Chuẩn hoá log-spectrogram ---
    log_spectrogram = librosa.amplitude_to_db(magnitude_model, ref=np.max)
    min_val = np.min(log_spectrogram)
    max_val = np.max(log_spectrogram)
    norm_spectrogram = (log_spectrogram - min_val) / (max_val - min_val + 1e-8)

    time_frames = norm_spectrogram.shape[1]
    window_size = 128
    step_size = 64   # overlap 50% → đủ mượt, ít chunk hơn

    if time_frames < window_size:
        num_chunks = 1
        padded_time_frames = window_size
    else:
        num_chunks = int(np.ceil((time_frames - window_size) / step_size)) + 1
        padded_time_frames = (num_chunks - 1) * step_size + window_size

    pad_len = padded_time_frames - time_frames
    if pad_len > 0:
        norm_spectrogram = np.pad(norm_spectrogram, ((0, 0), (0, pad_len)), mode='edge')

    # --- Tạo Hann window trên chiều time để crossfade khi overlap-add ---
    hann_win = np.hanning(window_size)  # shape: (128,)

    chunks = []
    start_indices = []
    for i in range(num_chunks):
        start = i * step_size
        chunk = norm_spectrogram[:, start: start + window_size]
        chunks.append(chunk)
        start_indices.append(start)

    X_input = np.array(chunks)[..., np.newaxis]  # (N, 128, 128, 1)

    del chunks, norm_spectrogram, log_spectrogram, magnitude_model
    gc.collect()

    # --- Chạy model ---
    X_cleaned = ai_model.predict(X_input, batch_size=8, verbose=0)
    X_cleaned = np.squeeze(X_cleaned, axis=-1)  # (N, 128, 128)

    del X_input
    gc.collect()

    # --- Overlap-add có crossfade (weighted OLA) ---
    cleaned_padded  = np.zeros((freq_bins, padded_time_frames))
    weight_padded   = np.zeros(padded_time_frames)   # chỉ cần 1D vì weight không phụ thuộc freq

    for i in range(num_chunks):
        start = start_indices[i]
        # Áp Hann window theo chiều time trước khi cộng
        cleaned_padded[:, start: start + window_size] += X_cleaned[i] * hann_win[np.newaxis, :]
        weight_padded[start: start + window_size]     += hann_win

    # Chia cho tổng trọng số (tránh chia cho 0 ở vùng biên)
    weight_padded = np.maximum(weight_padded, 1e-8)
    cleaned_padded /= weight_padded[np.newaxis, :]
    cleaned_spectrogram_norm = cleaned_padded[:, :time_frames]

    # --- Denormalise → biên độ ---
    cleaned_log_spectrogram = cleaned_spectrogram_norm * (max_val - min_val + 1e-8) + min_val
    cleaned_magnitude = librosa.db_to_amplitude(cleaned_log_spectrogram)  # (128, T)

    # Ghép lại với phần freq bins còn lại của stft gốc (nếu có) để istft đủ kích thước
    # magnitude gốc có shape (129, T); ta chỉ xử lý 128 bin đầu, bin 129 giữ nguyên
    full_magnitude = np.copy(magnitude[:, :time_frames])
    full_magnitude[:MODEL_FREQ_BINS, :] = cleaned_magnitude
    cleaned_stft = full_magnitude * phase[:, :time_frames]

    # --- Inverse STFT ---
    y_clean = librosa.istft(cleaned_stft, hop_length=HOP_LENGTH, win_length=N_FFT,
                            window='hann', length=len(y))

    # Normalise biên độ về 0.8 để tránh clipping
    max_amplitude = np.max(np.abs(y_clean))
    if max_amplitude > 0:
        y_clean = y_clean * (0.8 / max_amplitude)

    sf.write(output_path, y_clean, sr)

    del y, stft, phase, full_magnitude, cleaned_padded, cleaned_stft, X_cleaned
    gc.collect()

# ==========================================
# 4. API ENDPOINTS (FastAPI)
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "AI Audio Denoiser"}

# TUYỆT ĐỐI KHÔNG DÙNG "async def" Ở ĐÂY ĐỂ TRÁNH BLOCK MAIN THREAD
@app.post("/api/clean-audio")
def clean_audio_api(audio: UploadFile = File(...)):
    if not ai_model:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng")
    if not audio.filename:
        raise HTTPException(status_code=400, detail="File rỗng")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ext = os.path.splitext(audio.filename)[1].lower() or ".webm"
            input_temp_path = os.path.join(temp_dir, f'raw_input{ext}')
            wav_temp_path = os.path.join(temp_dir, 'converted_input.wav')
            output_temp_path = os.path.join(temp_dir, 'clean_output.wav')

            # Đọc file đồng bộ (thay cho await audio.read())
            with open(input_temp_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            print("🔄 Chuyển định dạng WebM -> WAV...", flush=True)
            audio_segment = AudioSegment.from_file(input_temp_path)
            audio_segment.export(wav_temp_path, format="wav")
            
            del audio_segment
            gc.collect()

            print("🎙️ Đang xử lý AI...", flush=True)
            process_audio(wav_temp_path, output_temp_path)
            print("✨ Xử lý thành công!", flush=True)
            
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
        print(f"❌ Lỗi: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=str(e))