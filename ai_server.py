import os
import tempfile
import io
import gc
import shutil
import time
import numpy as np
import threading
import uuid

# ==========================================
# 1. TỐI ƯU HÓA HỆ ĐIỀU HÀNH & TENSORFLOW
# ==========================================
os.environ['MALLOC_ARENA_MAX'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import librosa
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. TẢI MÔ HÌNH TOÀN CỤC VÀ KHÓA LUỒNG
# ==========================================
import tensorflow as tf

# Ép TensorFlow cấm tạo thêm luồng ẩn, chỉ dùng 1 luồng duy nhất để không chiếm đoạt CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
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

tasks_db = {}
ai_processing_lock = threading.Lock()

# ==========================================
# 3. HÀM XỬ LÝ ÂM THANH CỐT LÕI (BẢN VÁ CPU STARVATION)
# ==========================================
def process_audio(input_path, output_path, task_id):
    y, sr = librosa.load(input_path, sr=16000)
    
    SEGMENT_SECONDS = 5
    segment_samples = SEGMENT_SECONDS * sr
    y_clean_full = [] 

    total_segments = len(range(0, len(y), segment_samples))
    current_segment = 0

    for start_sample in range(0, len(y), segment_samples):
        current_segment += 1
        print(f"[{task_id}] ⚙️ Đang xử lý đoạn {current_segment}/{total_segments}...", flush=True)

        end_sample = min(start_sample + segment_samples, len(y))
        y_segment = y[start_sample:end_sample]
        
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

        X_cleaned = np.zeros((num_chunks, 128, 128), dtype=np.float32)
        # Giảm batch_size xuống 2 để AI nhai nhanh, nhả CPU lẹ
        batch_size = 2 

        # --- LÕI SÂU NHẤT CỦA AI ---
        for i in range(0, num_chunks, batch_size):
            end_idx = min(i + batch_size, num_chunks)
            batch = tf.convert_to_tensor(X_input[i:end_idx], dtype=tf.float32)
            
            # Tính toán khốc liệt bằng C++ Backend
            pred = ai_model(batch, training=False).numpy()
            X_cleaned[i:end_idx] = np.squeeze(pred, axis=-1)
            
            del batch, pred
            gc.collect()

            # --- CÚ CHỐT GIẢI CỨU CPU STARVATION ---
            # Ép luồng AI ngủ 0.05 giây SAU MỖI BATCH NHỎ ĐỂ FASTAPI CÓ THỂ THỞ VÀ TRẢ LỜI POLLING
            time.sleep(0.05) 
        # ---------------------------

        del X_input
        gc.collect()

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

        y_clean_segment = librosa.istft(cleaned_stft, hop_length=128, length=len(y_segment))
        y_clean_full.append(y_clean_segment)

        del stft, phase, cleaned_padded, overlap_count, cleaned_spectrogram_norm, cleaned_log_spectrogram, cleaned_magnitude, cleaned_stft, X_cleaned, y_clean_segment, y_segment
        gc.collect()

    y_final = np.concatenate(y_clean_full)
    
    max_amplitude = np.max(np.abs(y_final))
    if max_amplitude > 0:
        y_final = y_final * (0.8 / max_amplitude)

    sf.write(output_path, y_final, sr)

    del y, y_clean_full, y_final
    gc.collect()


def run_ai_background(task_id: str, input_path: str, wav_temp_path: str, output_temp_path: str):
    try:
        print(f"[{task_id}] ⏳ Đang xếp hàng chờ đến lượt xử lý...", flush=True)
        
        with ai_processing_lock:
            print(f"[{task_id}] 🟢 Đã đến lượt! Bắt đầu xử lý...", flush=True)
            
            audio_segment = AudioSegment.from_file(input_path)
            audio_segment.export(wav_temp_path, format="wav")
            del audio_segment
            gc.collect()

            process_audio(wav_temp_path, output_temp_path, task_id)
            
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result_file"] = output_temp_path
            
            print(f"[{task_id}] ✨ Đã hoàn thành! Mở khóa cho người tiếp theo.", flush=True)
            
    except Exception as e:
        import traceback
        print(f"[{task_id}] ❌ Lỗi: {e}", flush=True)
        tasks_db[task_id] = {"status": "failed", "error": str(e)}

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "AI Audio Denoiser"}

@app.post("/api/clean-audio")
def clean_audio_api(background_tasks: BackgroundTasks, audio: UploadFile = File(...)):
    if not ai_model:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng")
    if not audio.filename:
        raise HTTPException(status_code=400, detail="File rỗng")

    task_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    ext = os.path.splitext(audio.filename)[1].lower() or ".webm"
    input_temp_path = os.path.join(temp_dir, f'raw_input_{task_id}{ext}')
    wav_temp_path = os.path.join(temp_dir, f'converted_{task_id}.wav')
    output_temp_path = os.path.join(temp_dir, f'clean_{task_id}.wav')

    with open(input_temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    tasks_db[task_id] = {"status": "processing"}
    background_tasks.add_task(run_ai_background, task_id, input_temp_path, wav_temp_path, output_temp_path)

    return {"task_id": task_id, "status": "processing"}

@app.get("/api/task-status/{task_id}")
def get_task_status(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy task này")
    return {"task_id": task_id, "status": task["status"]}

@app.get("/api/download/{task_id}")
def download_result(task_id: str):
    task = tasks_db.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(status_code=400, detail="File chưa sẵn sàng")
    
    file_path = task["result_file"]
    
    with open(file_path, 'rb') as f:
        return_data = io.BytesIO(f.read())
    return_data.seek(0)
    
    del tasks_db[task_id]
    shutil.rmtree(os.path.dirname(file_path), ignore_errors=True)
    gc.collect()

    return StreamingResponse(
        return_data,
        media_type='audio/wav',
        headers={"Content-Disposition": f"attachment; filename=clean_{task_id}.wav"}
    )