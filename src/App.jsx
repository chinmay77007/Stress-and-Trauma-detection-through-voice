import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);

  const [partialText, setPartialText] = useState("");
  const [finalText, setFinalText] = useState("");

  const [svi, setSvi] = useState(null);
  const [textFeatures, setTextFeatures] = useState(null);
  const [voiceFeatures, setVoiceFeatures] = useState(null);

  const [error, setError] = useState("");

  const socketRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const workletNodeRef = useRef(null);
  const streamRef = useRef(null);

  // ============================
  // START RECORDING
  // ============================

  const startRecording = async () => {
    try {
      setError("");
      setPartialText("");
      setFinalText("");

      setSvi(null);
      setTextFeatures(null);
      setVoiceFeatures(null);

      // Microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // Flask WebSocket
      const socket = new WebSocket("ws://127.0.0.1:5000/transcribe");

      socket.binaryType = "arraybuffer";

      socket.onopen = async () => {
        console.log("WebSocket connected");

        try {
          await setupAudio(stream, socket);
          setRecording(true);
        } catch (audioSetupError) {
          console.error("Audio worklet setup failed:", audioSetupError);
          setError(
            `Audio setup failed: ${audioSetupError.message || audioSetupError}`
          );

          // Don't leave a half-open socket/mic hanging around
          stream.getTracks().forEach((track) => track.stop());
          if (socket.readyState === WebSocket.OPEN) {
            socket.close();
          }
        }
      };

      // ============================
      // RECEIVE DATA FROM FLASK
      // ============================

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // ----------------------------
          // PARTIAL TRANSCRIPTION
          // ----------------------------

          if (data.type === "partial") {
            setPartialText(data.text || "");
          }

          // ----------------------------
          // FINAL RESULT
          // ----------------------------

          if (data.type === "final") {
            console.log("Final result:", data);

            if (data.text) {
              setFinalText((previous) => {
                const combined = previous
                  ? `${previous} ${data.text}`
                  : data.text;

                return combined.trim();
              });
            }

            // Text features from spaCy
            if (data.features) {
              setTextFeatures((previous) => ({
                ...(previous || {}),
                ...data.features,
              }));
            }

            // Voice analysis
            if (data.voice) {
              setVoiceFeatures((previous) => ({
                ...(previous || {}),
                ...data.voice,
              }));
            }

            // SVI result
            if (data.svi) {
              setSvi(data.svi);
            }

            setPartialText("");
          }

          // ----------------------------
          // ERROR
          // ----------------------------

          if (data.type === "error") {
            console.error("Backend error:", data.message);
            setError(data.message || "Backend error");
          }
        } catch (err) {
          console.error("Invalid WebSocket message:", err);
        }
      };

      socket.onerror = () => {
        setError("Could not connect to the Flask server.");
      };

      socket.onclose = () => {
        console.log("WebSocket closed");
      };

      socketRef.current = socket;
    } catch (err) {
      console.error(err);

      setError(
        "Microphone access failed. Please allow microphone permission."
      );
    }
  };

  // ============================
  // AUDIO SETUP (AudioWorkletNode)
  // ============================
  //
  // Replaces the deprecated ScriptProcessorNode. The heavy lifting --
  // adaptive VAD gating and 16kHz downsampling -- happens inside
  // pcm-worklet-processor.js, on the dedicated audio rendering thread.
  // This function just wires up the graph and forwards whatever PCM the
  // worklet decides is speech.

  const setupAudio = async (stream, socket) => {
    const audioContext = new AudioContext();
    audioContextRef.current = audioContext;

    console.log("Browser sample rate:", audioContext.sampleRate);

    // Must be served as a static file -- for Create React App / Vite,
    // place pcm-worklet-processor.js in the `public/` folder so it's
    // reachable at this root-relative path.
    try {
      await audioContext.audioWorklet.addModule("/pcm-worklet-processor.js");
    } catch (err) {
      throw new Error(
        `Could not load pcm-worklet-processor.js (check it's in your ` +
          `public/ folder and reachable at that URL): ${err.message || err}`
      );
    }

    const source = audioContext.createMediaStreamSource(stream);
    sourceRef.current = source;

    // ----------------------------
    // HIGH PASS FILTER
    // ----------------------------

    const highpass = audioContext.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 80;
    highpass.Q.value = 0.7;

    // ----------------------------
    // LOW PASS FILTER
    // ----------------------------

    const lowpass = audioContext.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 7500;
    lowpass.Q.value = 0.7;

    // ----------------------------
    // AUDIO WORKLET NODE
    // ----------------------------

    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    workletNodeRef.current = workletNode;

    workletNode.port.onmessage = (event) => {
      if (
        event.data?.type === "audio" &&
        socket.readyState === WebSocket.OPEN
      ) {
        socket.send(event.data.buffer);
      }
    };

    // ----------------------------
    // AUDIO GRAPH
    // ----------------------------
    //
    // Microphone -> High-pass -> Low-pass -> Worklet (VAD + resample) -> WebSocket

    source.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(workletNode);

    // Some browsers require an output connection to keep a worklet node
    // running. We don't want audible mic feedback, so route through a
    // silent gain node.
    const silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    workletNode.connect(silentGain);
    silentGain.connect(audioContext.destination);
  };

  // ============================
  // STOP RECORDING
  // ============================

  const stopRecording = () => {
    setRecording(false);

    // Tell Flask to finalize
    if (
      socketRef.current &&
      socketRef.current.readyState === WebSocket.OPEN
    ) {
      socketRef.current.send(
        JSON.stringify({
          type: "stop",
        })
      );
    }

    // Stop microphone
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    // Disconnect worklet node
    if (workletNodeRef.current) {
      workletNodeRef.current.port.onmessage = null;
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Disconnect source
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Give Flask time to send final result
    setTimeout(() => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    }, 700);
  };

  // ============================
  // HELPER
  // ============================

  const formatValue = (value) => {
    if (value === null || value === undefined) {
      return "N/A";
    }

    if (typeof value === "number") {
      return Number.isInteger(value) ? value : value.toFixed(3);
    }

    if (typeof value === "object") {
      return JSON.stringify(value, null, 2);
    }

    return String(value);
  };

  // ============================
  // UI
  // ============================

  return (
    <div className="app">
      <h1>SIH Voice Assessment</h1>

      <p className="description">
        Speak into the microphone to analyze speech, language, and voice
        patterns.
      </p>

      {/* MICROPHONE */}

      <button
        className={recording ? "mic-button recording" : "mic-button"}
        onClick={recording ? stopRecording : startRecording}
      >
        {recording ? "⏹" : "🎤"}
      </button>

      <p className="status">
        {recording ? "Listening..." : "Click the microphone to start"}
      </p>

      {/* ERROR */}

      {error && <div className="error">{error}</div>}

      {/* ============================
          TRANSCRIPT
      ============================ */}

      <div className="transcript-box">
        <h2>Transcript</h2>

        <p className="final-text">{finalText}</p>

        {recording && partialText && (
          <p className="partial-text">{partialText}</p>
        )}

        {!finalText && !partialText && (
          <p className="placeholder">Start speaking...</p>
        )}
      </div>

      {/* ============================
          SVI RESULT
      ============================ */}

      {svi && (
        <div className="result-box">
          <h2>Speech Vulnerability Index</h2>

          <div className="svi-score">
            {formatValue(svi.svi_score)}
          </div>

          <div className="svi-category">{svi.risk_category}</div>

          <pre>{JSON.stringify(svi, null, 2)}</pre>
        </div>
      )}

      {/* ============================
          TEXT FEATURES
      ============================ */}

      {textFeatures && (
        <div className="features-box">
          <h2>Text Analysis</h2>

          <pre>{JSON.stringify(textFeatures, null, 2)}</pre>
        </div>
      )}

      {/* ============================
          VOICE FEATURES
      ============================ */}

      {voiceFeatures && (
        <div className="features-box">
          <h2>Voice Analysis</h2>

          <pre>{JSON.stringify(voiceFeatures, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
