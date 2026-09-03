import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      setError("");
      setTranscript("");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        // Stop microphone
        stream.getTracks().forEach((track) => track.stop());

        await sendAudioToBackend(audioBlob);
      };

      mediaRecorder.start();

      setRecording(true);
    } catch (err) {
      console.error(err);

      setError(
        "Could not access microphone. Please allow microphone permission."
      );
    }
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setProcessing(true);
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    const formData = new FormData();

    formData.append("audio", audioBlob, "recording.webm");

    try {
      const response = await fetch("http://127.0.0.1:5000/transcribe", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Transcription failed");
      }

      setTranscript(data.text);
    } catch (err) {
      console.error(err);

      setError(
        "Could not transcribe audio. Make sure the Flask server is running."
      );
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="app">
      <h1>SIH Voice Input</h1>

      <p className="description">
        Speak into your microphone and your speech will be converted to text.
      </p>

      <button
        className={`mic-button ${recording ? "recording" : ""}`}
        onClick={recording ? stopRecording : startRecording}
      >
        {recording ? "⏹" : "🎤"}
      </button>

      <p className="status">
        {recording
          ? "Listening... Click to stop"
          : processing
          ? "Processing audio..."
          : "Click the microphone to start"}
      </p>

      {error && <div className="error">{error}</div>}

      <textarea
        value={transcript}
        onChange={(e) => setTranscript(e.target.value)}
        placeholder="Your transcript will appear here..."
      />
    </div>
  );
}

export default App;
