from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import whisper

app = FastAPI(title="Whisper Audio Service")

model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

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
        # transcription (langue auto)
        result = model.transcribe(tmp_path)

        detected_language = result.get("language")
        text_original = (result.get("text") or "").strip()

        # traduction vers anglais (task=translate)
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
