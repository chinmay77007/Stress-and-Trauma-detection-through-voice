import re
import spacy


# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")


# Words/phrases useful for a first-pass distress indicator.
# These are NOT diagnoses. Kept alongside the ML model below because they're
# fully interpretable (you can show exactly which words triggered a score),
# which matters for a government-facing tool -- but they're brittle on
# their own (paraphrased distress with none of these exact words scores 0),
# so they're now a secondary signal rather than the primary one.
FEAR_WORDS = {
    "scared", "afraid", "fear", "terrified", "frightened", "panic",
    "panicking", "horrible", "danger", "dangerous", "threat", "threatened",
}

DISTRESS_WORDS = {
    "cry", "crying", "sad", "upset", "helpless", "hopeless", "broken",
    "hurt", "pain", "distressed", "shaking", "trauma",
}

ANXIETY_WORDS = {
    "worried", "worry", "worrying", "anxious", "anxiety", "nervous",
    "uneasy", "restless", "stress", "stressed",
}

UNCERTAINTY_WORDS = {
    "don't know", "dont know", "not sure", "unsure", "maybe", "perhaps",
    "uncertain", "confused", "confusion",
}

SLEEP_DISTRESS_WORDS = {
    "can't sleep", "cant sleep", "cannot sleep", "couldn't sleep",
    "couldnt sleep", "insomnia", "sleep", "sleepless",
}

HESITATION_WORDS = {
    "um", "uh", "umm", "uhh", "er", "hmm",
}


# --- Pretrained emotion classifier (primary distress signal) ---
_emotion_pipeline = None
_EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
_MAX_CHARS_FOR_MODEL = 1000  # defensive truncation; the model has a token limit


def _get_emotion_pipeline():
    global _emotion_pipeline

    if _emotion_pipeline is not None:
        return _emotion_pipeline

    try:
        from transformers import pipeline

        _emotion_pipeline = pipeline(
            "text-classification",
            model=_EMOTION_MODEL_NAME,
            top_k=None,  # return probabilities for every emotion class
        )
    except Exception as e:
        print(f"[text_features] Emotion model unavailable, falling back to "
              f"keyword-only scoring: {e}")
        _emotion_pipeline = False  # sentinel: "tried and failed"

    return _emotion_pipeline


def run_emotion_model(text):
    """
    Run the pretrained emotion classifier on raw (non-normalized) text.
    Returns {"emotion_probs": {...}, "ml_distress": 0-1} or None if the
    model isn't available.
    """
    pipe = _get_emotion_pipeline()

    if pipe is False or pipe is None:
        return None

    if not text or not text.strip():
        return None

    try:
        results = pipe(text[:_MAX_CHARS_FOR_MODEL])

        # With top_k=None, output is [[{label, score}, ...]]
        if isinstance(results[0], list):
            results = results[0]

        probs = {r["label"].lower(): float(r["score"]) for r in results}
    except Exception as e:
        print(f"[text_features] Emotion inference failed: {e}")
        return None

    # Distress-correlated emotion classes from this model's label set
    ml_distress = (
        probs.get("fear", 0.0)
        + probs.get("sadness", 0.0)
        + probs.get("anger", 0.0)
        + probs.get("disgust", 0.0)
    )
    ml_distress = float(min(ml_distress, 1.0))

    return {
        "emotion_probs": probs,
        "ml_distress": ml_distress,
    }


def normalize_text(text):
    """Basic text normalization for keyword matching (lowercase, whitespace)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_word_matches(text, word_set):
    """Count how many words/phrases from word_set occur in the transcript."""
    count = 0

    for word in word_set:
        if " " in word:
            count += text.count(word)
        else:
            pattern = rf"\b{re.escape(word)}\b"
            count += len(re.findall(pattern, text))

    return count


def calculate_density(count, total_words):
    """Converts raw count into a normalized 0-1 score."""
    if total_words == 0:
        return 0.0

    density = count / total_words
    return min(density * 10, 1.0)


def extract_text_features(text):
    """
    Extract linguistic features from a transcript, combining a pretrained
    emotion classifier (primary signal, catches paraphrased distress) with
    keyword density scoring (secondary/interpretable signal).
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "fear": 0.0,
            "distress": 0.0,
            "anxiety": 0.0,
            "uncertainty": 0.0,
            "sleep_distress": 0.0,
            "hesitation": 0.0,
            "negative_language": 0.0,
            "emotion_probs": None,
            "ml_distress": None,
        }

    normalized = normalize_text(text)
    doc = nlp(normalized)

    words = [
        token for token in doc
        if not token.is_punct and not token.is_space
    ]

    word_count = len(words)
    sentence_count = len(list(doc.sents))

    fear_count = count_word_matches(normalized, FEAR_WORDS)
    distress_count = count_word_matches(normalized, DISTRESS_WORDS)
    anxiety_count = count_word_matches(normalized, ANXIETY_WORDS)
    uncertainty_count = count_word_matches(normalized, UNCERTAINTY_WORDS)
    sleep_count = count_word_matches(normalized, SLEEP_DISTRESS_WORDS)
    hesitation_count = count_word_matches(normalized, HESITATION_WORDS)

    negative_count = fear_count + distress_count + anxiety_count
    keyword_negative_language = calculate_density(negative_count, word_count)

    # Run the pretrained emotion model on the ORIGINAL (non-lowercased) text --
    # transformer models generally use casing as a signal.
    emotion_result = run_emotion_model(text)

    if emotion_result is not None:
        # Blend: ML signal weighted higher since it generalizes beyond exact
        # keyword matches, but keyword density still contributes so an
        # emphatic keyword-heavy transcript isn't undercounted if the model
        # is uncertain.
        negative_language = float(
            0.4 * keyword_negative_language + 0.6 * emotion_result["ml_distress"]
        )
        emotion_probs = emotion_result["emotion_probs"]
        ml_distress = emotion_result["ml_distress"]
    else:
        negative_language = keyword_negative_language
        emotion_probs = None
        ml_distress = None

    features = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "fear": calculate_density(fear_count, word_count),
        "distress": calculate_density(distress_count, word_count),
        "anxiety": calculate_density(anxiety_count, word_count),
        "uncertainty": calculate_density(uncertainty_count, word_count),
        "sleep_distress": min(sleep_count / 2, 1.0),
        "hesitation": calculate_density(hesitation_count, word_count),
        "negative_language": negative_language,
        "emotion_probs": emotion_probs,
        "ml_distress": ml_distress,
    }

    return features
