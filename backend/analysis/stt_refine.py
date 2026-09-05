"""
Accuracy-boosting second pass for the FINAL transcript of each utterance.

Vosk stays as the low-latency streaming engine for live partial captions --
it's cheap and responsive, which is what partials need. But Vosk's accuracy
on Indian-accented English is weak (the largest available Indian-English
model, vosk-model-en-in-0.5, has a documented WER around 36% on Indian
English benchmarks). Since the transcript directly feeds the text-based
SVI signal, it's worth re-transcribing each finalized utterance with a
much more accurate model before scoring it.

faster-whisper (CTranslate2) is used here because it runs well on CPU with
int8 quantization -- no GPU required for a hackathon demo -- and it reuses
the SAME audio buffer already captured for voice_features.py, so there's no
extra audio plumbing needed.

Trade-off: this adds roughly 200ms-1.5s of extra latency per utterance
*after* the Vosk final fires (model-size dependent), in exchange for a
meaningfully more accurate transcript. Since it only runs once per
finalized utterance (not per audio chunk), this is an acceptable trade for
correctness-sensitive downstream scoring.
"""

import numpy as np

_whisper_model = None

# tiny.en: fastest, lowest accuracy | base.en: good balance | small.en: best
# accuracy of the .en family, still CPU-feasible with int8. Start with
# base.en and move to small.en if your machine keeps up.
WHISPER_MODEL_SIZE = "base.en"
SAMPLE_RATE = 16000


def _get_whisper_model():
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        # int8 on CPU keeps this fast enough for near-real-time use.
        # Switch to device="cuda", compute_type="float16" if you have a GPU.
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    except Exception as e:
        print(f"[stt_refine] faster-whisper unavailable, keeping Vosk "
              f"transcript only: {e}")
        _whisper_model = False  # sentinel: "tried and failed"

    return _whisper_model


def refine_transcript(pcm_bytes, vosk_text, sample_rate=SAMPLE_RATE):
    """
    Re-transcribe one utterance's raw Int16 PCM audio with Whisper for a
    more accurate final transcript. Falls back to the original Vosk text
    if Whisper isn't installed/available or produces nothing usable, so
    this is always safe to call.
    """
    model = _get_whisper_model()

    if model is False or model is None:
        return vosk_text

    if not pcm_bytes or len(pcm_bytes) < sample_rate * 0.3:
        # Too short a clip for Whisper to do anything useful with
        return vosk_text

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,  # trims leading/trailing silence internally
        )

        whisper_text = " ".join(seg.text.strip() for seg in segments).strip()

        return whisper_text if whisper_text else vosk_text

    except Exception as e:
        print(f"[stt_refine] Whisper inference failed, using Vosk text: {e}")
        return vosk_text
