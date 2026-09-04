import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);
  const [partialText, setPartialText] = useState("");
  const [finalText, setFinalText] = useState("");
  const [error, setError] = useState("");

  const socketRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  const startRecording = async () => {
    try {
      setError("");
      setPartialText("");
      setFinalText("");

      // Ask for microphone permission
    const stream =
      await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
      streamRef.current = stream;

      // Connect to Flask WebSocket
      const socket = new WebSocket(
        "ws://127.0.0.1:5000/transcribe"
      );

      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        console.log("WebSocket connected");

        setupAudio(stream, socket);

        setRecording(true);
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "partial") {
          setPartialText(data.text);
        }

        if (data.type === "final") {
          if (data.text) {
            setFinalText((previous) => {
              const combined = previous
                ? `${previous} ${data.text}`
                : data.text;

              return combined.trim();
            });
          }

          setPartialText("");
        }

        if (data.type === "error") {
          setError(data.message);
        }
      };

      socket.onerror = () => {
        setError(
          "Could not connect to the Flask server."
        );
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


	const setupAudio = (stream, socket) => {
	  const audioContext = new AudioContext();

	  audioContextRef.current = audioContext;

	  console.log(
	    "Browser sample rate:",
	    audioContext.sampleRate
	  );

	  const source =
	    audioContext.createMediaStreamSource(stream);

	  sourceRef.current = source;

	  /*
	   * ============================
	   * 1. HIGH-PASS FILTER
	   * ============================
	   *
	   * Removes low-frequency noise:
	   * fans, AC, desk vibrations, etc.
	   */

	  const highpass =
	    audioContext.createBiquadFilter();

	  highpass.type = "highpass";
	  highpass.frequency.value = 80;
	  highpass.Q.value = 0.7;

	  /*
	   * ============================
	   * 2. LOW-PASS FILTER
	   * ============================
	   *
	   * Removes unnecessary high-frequency
	   * microphone noise.
	   */

	  const lowpass =
	    audioContext.createBiquadFilter();

	  lowpass.type = "lowpass";
	  lowpass.frequency.value = 7500;
	  lowpass.Q.value = 0.7;

	  /*
	   * ============================
	   * 3. SCRIPT PROCESSOR
	   * ============================
	   */

	  const processor =
	    audioContext.createScriptProcessor(
	      4096,
	      1,
	      1
	    );

	  processorRef.current = processor;

	  /*
	   * ============================
	   * 4. AUDIO PROCESSING
	   * ============================
	   */

	  processor.onaudioprocess = (event) => {
	    if (
	      socket.readyState !==
	      WebSocket.OPEN
	    ) {
	      return;
	    }

	    const input =
	      event.inputBuffer.getChannelData(0);

	    /*
	     * Calculate RMS volume.
	     *
	     * This lets us detect silence.
	     */

	    let sum = 0;

	    for (let i = 0; i < input.length; i++) {
	      sum += input[i] * input[i];
	    }

	    const rms =
	      Math.sqrt(sum / input.length);

	    /*
	     * Ignore extremely quiet audio.
	     *
	     * This prevents sending pure silence
	     * and some background noise to Vosk.
	     */

	    if (rms < 0.008) {
	      return;
	    }

	    /*
	     * ============================
	     * 5. RESAMPLE
	     * ============================
	     *
	     * Firefox:
	     *
	     * 48000 Hz
	     *
	     * Vosk:
	     *
	     * 16000 Hz
	     */

	    const inputSampleRate =
	      audioContext.sampleRate;

	    const outputSampleRate = 16000;

	    const ratio =
	      inputSampleRate /
	      outputSampleRate;

	    const outputLength =
	      Math.floor(
		input.length / ratio
	      );

	    const pcm =
	      new Int16Array(outputLength);

	    /*
	     * ============================
	     * 6. DOWNSAMPLE
	     * ============================
	     */

	    for (
	      let i = 0;
	      i < outputLength;
	      i++
	    ) {
	      const position =
		i * ratio;

	      const index =
		Math.floor(position);

	      const nextIndex =
		Math.min(
		  index + 1,
		  input.length - 1
		);

	      const fraction =
		position - index;

	      /*
	       * Linear interpolation
	       */

	      const sample =
		input[index] *
		  (1 - fraction) +
		input[nextIndex] *
		  fraction;

	      /*
	       * Clamp to [-1, 1]
	       */

	      const clamped =
		Math.max(
		  -1,
		  Math.min(1, sample)
		);

	      /*
	       * Float32 -> Int16
	       */

	      pcm[i] =
		clamped < 0
		  ? clamped * 0x8000
		  : clamped * 0x7fff;
	    }

	    socket.send(
	      pcm.buffer
	    );
	  };

	  /*
	   * ============================
	   * AUDIO GRAPH
	   * ============================
	   *
	   * Microphone
	   *     ↓
	   * High-pass
	   *     ↓
	   * Low-pass
	   *     ↓
	   * Processor
	   *     ↓
	   * WebSocket
	   */

	  source.connect(highpass);

	  highpass.connect(lowpass);

	  lowpass.connect(processor);

	  /*
	   * Silent output node.
	   *
	   * Keeps ScriptProcessor active without
	   * producing microphone feedback.
	   */

	  const silentGain =
	    audioContext.createGain();

	  silentGain.gain.value = 0;

	  processor.connect(
	    silentGain
	  );

	  silentGain.connect(
	    audioContext.destination
	  );
	};
  const stopRecording = () => {

    setRecording(false);

    // Tell Flask to finalize recognition
    if (
      socketRef.current &&
      socketRef.current.readyState ===
        WebSocket.OPEN
    ) {

      socketRef.current.send(
        JSON.stringify({
          type: "stop",
        })
      );
    }

    // Stop microphone
    if (streamRef.current) {

      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    // Disconnect audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (audioContextRef.current) {

      audioContextRef.current.close();

      audioContextRef.current = null;
    }

    // Close socket after server has received stop
    setTimeout(() => {

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }

    }, 500);
  };


  return (
    <div className="app">

      <h1>SIH Voice Assessment</h1>

      <p className="description">
        Speak into the microphone. Your speech
        will appear in real time.
      </p>

      <button
        className={
          recording
            ? "mic-button recording"
            : "mic-button"
        }
        onClick={
          recording
            ? stopRecording
            : startRecording
        }
      >
        {recording ? "⏹" : "🎤"}
      </button>

      <p className="status">

        {recording
          ? "Listening..."
          : "Click the microphone to start"}

      </p>


      {error && (
        <div className="error">
          {error}
        </div>
      )}


      <div className="transcript-box">

        <h2>Transcript</h2>

        <p className="final-text">
          {finalText}
        </p>

        {recording && partialText && (
          <p className="partial-text">
            {partialText}
          </p>
        )}

        {!finalText && !partialText && (
          <p className="placeholder">
            Start speaking...
          </p>
        )}

      </div>

    </div>
  );
}

export default App;
