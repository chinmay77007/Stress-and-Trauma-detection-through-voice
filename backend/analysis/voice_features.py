"""
Voice analysis module.

Two complementary signals are extracted from each utterance's raw audio:

1. Prosodic features (pitch variance, energy variance, pause ratio, speaking
   rate) computed directly with librosa. These are language-independent and
   fully interpretable -- useful both as a fallback and for explaining *why*
   the system flagged someone (important for a government-facing tool).

2. A pretrained speech-emotion-recognition (SER) model (wav2vec2-based) that
   outputs emotion probabilities directly from the waveform.

NOTE: this is a hackathon prototype. Prosodic thresholds below are based on
general findings in affective-computing literature (elevated pitch variance,
reduced pause regularity, faster/pressured speech correlate with distress/
arousal) -- they are heuristic, not clinically validated. Do not present the
output as a diagnosis; it is a triage signal for a human to review.
"""

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Lazy-loaded pretrained SER pipeline (loaded once, on first use)
_ser_pipeline = None
_SER_MODEL_NAME = "superb/wav2vec2-base-superb-er"  # labels: neu, hap, ang, sad
SAMPLE_RATE = 16000


def _get_ser_pipeline():
    global _ser_pipeline

    if _ser_pipeline is not None:
        return _ser_pipeline

    try:
        from transformers import pipeline

        _ser_pipeline = pipeline(
            "audio-classification",
            model=_SER_MODEL_NAME,
        )
    except Exception as e:
        print(f"[voice_features] SER model unavailable, falling back "
              f"to prosody-only scoring: {e}")
        _ser_pipeline = False  # sentinel: "tried and failed"

    return _ser_pipeline


def pcm_int16_to_float32(pcm_bytes):
    """Convert raw Int16 PCM bytes (as sent by the frontend) to float32 [-1, 1]."""
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def extract_prosodic_features(audio, sample_rate=SAMPLE_RATE):
    """
    Compute interpretable prosodic features from a float32 audio buffer.
    Returns raw feature values plus a normalized 0-1 'prosodic_distress' score.
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is required: pip install librosa")

    if audio is None or len(audio) < sample_rate * 0.3:
        # Too short to extract meaningful prosody
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "pause_ratio": 0.0,
            "speaking_rate_proxy": 0.0,
            "prosodic_distress": 0.0,
        }

    # --- Pitch (F0) via YIN ---
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    voiced_f0 = f0[f0 > 0]
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # --- Energy (RMS) ---
    rms = librosa.feature.rms(y=audio)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # --- Pauses: fraction of frames below an energy threshold ---
    silence_threshold = energy_mean * 0.25
    pause_ratio = float(np.mean(rms < silence_threshold))

    # --- Speaking rate proxy: voiced-frame ratio (rough stand-in without ASR alignment) ---
    speaking_rate_proxy = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0

    # --- Heuristic distress score ---
    # Normalize each sub-signal against rough "typical calm speech" reference
    # ranges, then average. These reference values are approximate and should
    # be tuned against your own pilot recordings if possible.
    pitch_var_score = min(pitch_std / 60.0, 1.0)          # high pitch variance -> arousal
    energy_var_score = min(energy_std / (energy_mean + 1e-6) / 1.5, 1.0)  # unstable loudness
    pause_score = min(pause_ratio / 0.6, 1.0)              # excessive pausing -> hesitation/distress

    prosodic_distress = float(
        np.clip((pitch_var_score + energy_var_score + pause_score) / 3.0, 0.0, 1.0)
    )

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "pause_ratio": pause_ratio,
        "speaking_rate_proxy": speaking_rate_proxy,
        "prosodic_distress": prosodic_distress,
    }


def run_ser_model(audio, sample_rate=SAMPLE_RATE):
    """
    Run the pretrained SER model on a float32 audio buffer.
    Returns emotion probabilities and a 0-1 'ser_distress' score, or None
    if the model isn't available (caller should fall back to prosody only).
    """
    pipe = _get_ser_pipeline()

    if pipe is False or pipe is None:
        return None

    if audio is None or len(audio) < sample_rate * 0.5:
        return None

    try:
        results = pipe({"array": audio, "sampling_rate": sample_rate})
    except Exception as e:
        print(f"[voice_features] SER inference failed: {e}")
        return None

    probs = {r["label"].lower(): float(r["score"]) for r in results}

    # This model's label set is {neu, hap, ang, sad} -- no explicit "fear" class.
    # Treat sadness + anger as the primary distress-correlated emotions.
    ser_distress = probs.get("sad", 0.0) + probs.get("ang", probs.get("angry", 0.0))
    ser_distress = float(np.clip(ser_distress, 0.0, 1.0))

    return {
        "emotion_probs": probs,
        "ser_distress": ser_distress,
    }


def analyze_voice(pcm_bytes, sample_rate=SAMPLE_RATE):
    """
    Main entry point: takes raw Int16 PCM bytes for one utterance and
    returns combined prosodic + SER analysis with a single 'voice_score'.
    """
    audio = pcm_int16_to_float32(pcm_bytes)

    prosodic = extract_prosodic_features(audio, sample_rate)
    ser = run_ser_model(audio, sample_rate)

    if ser is not None:
        # Weight the learned SER signal a bit higher than the hand-built
        # prosodic heuristic, but keep both -- prosody adds robustness if
        # the SER model (trained on acted English emotional speech) doesn't
        # transfer well to the actual call audio/language.
        voice_score = 0.6 * ser["ser_distress"] + 0.4 * prosodic["prosodic_distress"]
    else:
        voice_score = prosodic["prosodic_distress"]

    return {
        "prosodic": prosodic,
        "ser": ser,
        "voice_score": float(np.clip(voice_score, 0.0, 1.0)),
    }
