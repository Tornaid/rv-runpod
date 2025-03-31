import requests
import tempfile
import os
from faster_whisper import WhisperModel

def run(job):
    audio_url = job["input"].get("audio")
    if not audio_url:
        raise ValueError("Aucune URL audio fournie.")

    print(f"🎧 Téléchargement depuis Supabase : {audio_url}")

    try:
        # Téléchargement dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            response = requests.get(audio_url)
            response.raise_for_status()
            tmp_file.write(response.content)
            audio_path = tmp_file.name

        print(f"✅ Audio local : {audio_path}")

        # Transcription
        model = WhisperModel("medium")  # change si tu veux small/large
        segments, info = model.transcribe(audio_path)
        transcription = " ".join([s.text for s in segments])

        os.remove(audio_path)

        return {
            "transcription": transcription,
            "language": info.language
        }

    except Exception as e:
        print("❌ Erreur :", str(e))
        return { "error": str(e) }
