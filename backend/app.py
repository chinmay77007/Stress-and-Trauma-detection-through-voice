from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile

app = Flask(__name__)
CORS(app)

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded!")


@app.route("/transcribe", methods=["POST"])
def transcribe():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file received"
        }), 400

    audio = request.files["audio"]

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".webm"
    ) as temp:

        audio.save(temp.name)
        temp_path = temp.name

    try:

        # Transcribe audio
        result = model.transcribe(
            temp_path,
            language="en"
        )

        transcript = result["text"].strip()

        return jsonify({
            "success": True,
            "text": transcript
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/")
def home():
    return jsonify({
        "message": "SIH Speech Recognition API is running"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
