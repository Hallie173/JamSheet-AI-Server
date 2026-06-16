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
    print("⏳ Đang tải mô hình AI vào RAM...", flush=True)
    ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    ai_model.predict(np.zeros((1, 128, 128, 1), dtype=np.float32), verbose=0)
    print("✅ Mô hình đã sẵn sàng!", flush=True)
except Exception as e:
    print(f"❌ Lỗi tải mô hình: {e}", flush=True)
    ai_model = None

# ==========================================
# 3. HÀM XỬ LÝ ÂM THANH CỐT LÕI (CHIA ĐỂ TRỊ)
# ==========================================
def process_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)
    
    # CẮT DÒNG THỜI GIAN: Chia file thành các block 10 giây để chống tràn RAM
    SEGMENT_SECONDS = 10
    segment_samples = SEGMENT_SECONDS * sr
    
    y_clean_full = [] # Mảng chứa các mảnh âm thanh đã lọc sạch

    # Vòng lặp xử lý từng đoạn 10 giây
    for start_sample in range(0, len(y), segment_samples):
        end_sample = min(start_sample + segment_samples, len(y))
        y_segment = y[start_sample:end_sample]
        
        # --- TIỀN XỬ LÝ CHO ĐOẠN 10 GIÂY ---
        stft = librosa.stft(y_segment, n_fft=254, hop_length=128)
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

        del chunks, norm_spectrogram, log_spectrogram, magnitude
        gc.collect()

        # --- LỌC AI CHO ĐOẠN 10 GIÂY ---
        X_cleaned = np.zeros((num_chunks, 128, 128), dtype=np.float32)
        batch_size = 4 

        for i in range(0, num_chunks, batch_size):
            end_idx = min(i + batch_size, num_chunks)
            batch = tf.convert_to_tensor(X_input[i:end_idx], dtype=tf.float32)
            pred = ai_model(batch, training=False).numpy()
            X_cleaned[i:end_idx] = np.squeeze(pred, axis=-1)
            del batch, pred
            gc.collect()

        del X_input
        gc.collect()

        # --- HẬU XỬ LÝ (RÁP NỐI STFT) CHO ĐOẠN 10 GIÂY ---
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

        # Chuyển về sóng âm và đưa vào mảng lưu trữ tổng
        y_clean_segment = librosa.istft(cleaned_stft, hop_length=128, length=len(y_segment))
        y_clean_full.append(y_clean_segment)

        # Dọn sạch sành sanh mọi dữ liệu của block 10s này để đón block mới
        del stft, phase, cleaned_padded, overlap_count, cleaned_spectrogram_norm, cleaned_log_spectrogram, cleaned_magnitude, cleaned_stft, X_cleaned, y_clean_segment, y_segment
        gc.collect()

    # --- NỐI TẤT CẢ CÁC ĐOẠN LẠI THÀNH FILE HOÀN CHỈNH ---
    y_final = np.concatenate(y_clean_full)
    
    max_amplitude = np.max(np.abs(y_final))
    if max_amplitude > 0:
        y_final = y_final * (0.8 / max_amplitude)

    sf.write(output_path, y_final, sr)

    del y, y_clean_full, y_final
    gc.collect()


# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "AI Audio Denoiser"}

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