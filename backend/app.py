import json
import os
import spacy

from flask import Flask, jsonify
from flask_cors import CORS
from flask_sock import Sock
from vosk import Model, KaldiRecognizer


app = Flask(__name__)
CORS(app)

sock = Sock(app)


MODEL_PATH = "vosk-model-hi-0.22"
#MODEL_PATH = "vosk-model-en-in-0.5"
print("Loading Vosk model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Vosk model not found: {MODEL_PATH}"
    )

model = Model(MODEL_PATH)

print("Vosk model loaded!")

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded!")
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "SIH Speech Recognition API"
    })

def process_text(text):

    doc = nlp(text)

    tokens = []
    lemmas = []

    for token in doc:

        if not token.is_stop and not token.is_punct:

            tokens.append(token.text)
            lemmas.append(token.lemma_)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "sentence_count": len(list(doc.sents))
    }

@sock.route("/transcribe")
def transcribe(ws):

    print("Client connected")

    # Browser audio will be converted to
    # 16-bit mono PCM at 16000 Hz.
    recognizer = KaldiRecognizer(
        model,
        16000
    )

    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)

    try:

        while True:

            data = ws.receive()

            # Client closed connection
            if data is None:
                break

            # JSON messages are used for control
            if isinstance(data, str):

                message = json.loads(data)

                if message.get("type") == "stop":

                    final_result = json.loads(
                        recognizer.FinalResult()
                    )

                    ws.send(json.dumps({
                        "type": "final",
                        "text": final_result.get(
                            "text",
                            ""
                        )
                    }))

                    break

                continue

            # Audio data
            if recognizer.AcceptWaveform(data):

               result = json.loads(
                  recognizer.Result()
               )

               text = result.get("text", "")

               nlp_result = process_text(text)

               ws.send(json.dumps({
               "type": "final",
               "text": text,
               "nlp": nlp_result
            }))

            else:

                partial = json.loads(
                    recognizer.PartialResult()
                )

                ws.send(json.dumps({
                    "type": "partial",
                    "text": partial.get(
                        "partial",
                        ""
                    )
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


if __name__ == "__main__":

    print(
        "Starting server on "
        "http://127.0.0.1:5000"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
