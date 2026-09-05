import json
import os
import spacy

from flask import Flask, jsonify
from flask_cors import CORS
from flask_sock import Sock
from vosk import Model, KaldiRecognizer

from analysis.text_features import extract_text_features
from analysis.voice_features import analyze_voice
from analysis.svi_fusion import compute_svi
from analysis.stt_refine import refine_transcript

app = Flask(__name__)
CORS(app)

sock = Sock(app)

MODEL_PATH = "vosk-model-en-in-0.5"
print("Loading Vosk model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found: {MODEL_PATH}")

model = Model(MODEL_PATH)
print("Vosk model loaded!")

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded!")


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "SIH Speech Recognition + SVI API"
    })


@sock.route("/transcribe")
def transcribe(ws):
    print("Client connected")

    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)

    # Per-utterance audio buffer (reset after each final)
    utterance_audio = bytearray()

    # Session-level state -- accumulated across the WHOLE call, so scoring
    # reflects the full conversation instead of just the last utterance.
    session_transcript_parts = []
    session_voice_scores = []
    peak_svi = {"svi_score": 0.0, "risk_category": "Low"}

    try:
        while True:
            data = ws.receive()

            if data is None:
                break

            # --- Control messages from frontend ---
            if isinstance(data, str):
                message = json.loads(data)

                if message.get("type") == "stop":
                    final_result = json.loads(recognizer.FinalResult())
                    vosk_text = final_result.get("text", "")

                    result_payload = _build_final_payload(
                        vosk_text,
                        utterance_audio,
                        session_transcript_parts,
                        session_voice_scores,
                        peak_svi,
                    )
                    ws.send(json.dumps(result_payload))
                    break

                continue

            # --- Audio data ---
            utterance_audio.extend(data)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                vosk_text = result.get("text", "")

                if vosk_text:
                    result_payload = _build_final_payload(
                        vosk_text,
                        utterance_audio,
                        session_transcript_parts,
                        session_voice_scores,
                        peak_svi,
                    )
                    ws.send(json.dumps(result_payload))

                # Reset the audio buffer -- this utterance is done
                utterance_audio = bytearray()

            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "")

                # Partials stay Vosk-only -- Whisper only runs on finals,
                # since re-running it on every chunk would add too much
                # latency for live captions.
                ws.send(json.dumps({
                    "type": "partial",
                    "text": text
                }))

    except Exception as e:
        print("WebSocket error:", e)

        try:
            ws.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except Exception:
            pass

    finally:
        print("Client disconnected")


def _build_final_payload(
    vosk_text,
    utterance_audio_bytes,
    session_transcript_parts,
    session_voice_scores,
    peak_svi,
):
    """
    Run Whisper refinement + per-utterance features for THIS utterance
    (kept for transparency/debugging), then fold it into the running
    session-level transcript and voice-score history to compute the
    session-wide SVI, which is what should actually be treated as the
    caller's current risk reading -- scoring each short utterance in
    isolation badly underrepresents distress that only shows up once
    enough context has accumulated.
    """
    refined_text = vosk_text

    if utterance_audio_bytes and len(utterance_audio_bytes) > 0:
        try:
            refined_text = refine_transcript(bytes(utterance_audio_bytes), vosk_text)
        except Exception as e:
            print(f"Transcript refinement failed, using Vosk text: {e}")

    # Per-utterance features -- useful for debugging/transparency, but NOT
    # what should drive the risk category on their own (see docstring).
    utterance_text_features = extract_text_features(refined_text) if refined_text else {}

    voice_result = None
    if utterance_audio_bytes and len(utterance_audio_bytes) > 0:
        try:
            voice_result = analyze_voice(bytes(utterance_audio_bytes))
        except Exception as e:
            print(f"Voice analysis failed, continuing with text only: {e}")

    # --- Fold into session-level running state ---
    if refined_text:
        session_transcript_parts.append(refined_text)

    if voice_result:
        session_voice_scores.append(voice_result["voice_score"])

    cumulative_text = " ".join(session_transcript_parts)
    cumulative_text_features = (
        extract_text_features(cumulative_text) if cumulative_text else {}
    )

    if session_voice_scores:
        avg_voice_score = sum(session_voice_scores) / len(session_voice_scores)
        max_voice_score = max(session_voice_scores)
        # Blend the running average (overall tone) with the max seen so far
        # -- a genuine distress spike shouldn't get diluted by calmer
        # speech around it.
        session_voice_result = {
            "voice_score": 0.5 * avg_voice_score + 0.5 * max_voice_score
        }
    else:
        session_voice_result = None

    session_svi = compute_svi(
        cumulative_text_features, session_voice_result, transcript=cumulative_text
    )

    # Track the worst point reached this call -- so a disclosure mid-call
    # isn't lost if the caller calms down afterward.
    if session_svi["svi_score"] > peak_svi.get("svi_score", 0.0):
        peak_svi.clear()
        peak_svi.update(session_svi)

    return {
        "type": "final",
        "text": refined_text,
        "vosk_text": vosk_text,
        "features": utterance_text_features,     # this utterance only (debug)
        "voice": voice_result,                    # this utterance only (debug)
        "svi": session_svi,                       # cumulative session SVI -- treat this as the current reading
        "peak_svi": dict(peak_svi),               # worst point reached so far this call
    }


if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
