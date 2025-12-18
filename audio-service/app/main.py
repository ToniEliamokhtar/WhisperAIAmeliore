from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import whisper
import uuid

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


app = FastAPI(title="Whisper Audio Service")

model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

SPECTRO_DIR = "/tmp/spectrograms"
os.makedirs(SPECTRO_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("WHISPER_MODEL", "base")}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        detected_language = result.get("language")
        text_original = (result.get("text") or "").strip()

        result_en = model.transcribe(tmp_path, task="translate")
        text_english = (result_en.get("text") or "").strip()

        return {
            "filename": file.filename,
            "language": detected_language,
            "text_original": text_original,
            "text_english": text_english,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _safe_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


@app.post("/spectrogram")
async def spectrogram(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # NOTE: support WAV (scipy.io.wavfile.read)
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
        tmp_audio.write(await file.read())
        audio_path = tmp_audio.name

    image_id = uuid.uuid4().hex
    image_name = f"{image_id}.png"
    out_png_path = os.path.join(SPECTRO_DIR, image_name)

    try:
        sr, data = wavfile.read(audio_path)

        # Mono si stéréo
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data.mean(axis=1)

        data = data.astype(np.float32)
        maxv = np.max(np.abs(data)) if data.size else 0.0
        if maxv > 0:
            data = data / maxv

        plt.figure(figsize=(10, 4))
        plt.specgram(data, Fs=sr, NFFT=1024, noverlap=512)
        plt.title("Spectrogramme")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.colorbar(label="Intensité (dB)")
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=160)
        plt.close()

        background_tasks.add_task(_safe_remove, audio_path)

        # IMPORTANT: on retourne une URL qui marche vraiment
        return JSONResponse(
            {
                "message": "Spectrogramme généré avec succès",
                "original_file": file.filename,
                "image_url": f"http://localhost:8000/spectrogram/{image_name}",
            }
        )

    except Exception as e:
        _safe_remove(audio_path)
        _safe_remove(out_png_path)
        raise HTTPException(status_code=500, detail=f"Spectrogram error: {str(e)}")


@app.get("/spectrogram/{image_name}")
def get_spectrogram(image_name: str):
    path = os.path.join(SPECTRO_DIR, image_name)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Spectrogram introuvable")

    return FileResponse(path, media_type="image/png", filename="spectrogram.png")
