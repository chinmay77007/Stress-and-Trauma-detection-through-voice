"""
Stress Vulnerability Index (SVI) fusion module.

Combines the text-analysis output (analysis.text_features) and voice-analysis
output (analysis.voice_features) into a single 0-100 SVI score, a risk
category, and a recommended action -- plus an independent safety-override
check for suicidal ideation / extreme severity that is never diluted by
averaging with the other signals.

IMPORTANT: this is a prototype triage aid, not a diagnostic instrument.
Every "Critical" or "High" output should route to a human (counsellor /
officer) for review, not trigger any fully automated action.
"""

import re

# --- Fusion weights (tune against pilot data if available) ---
TEXT_WEIGHT = 0.45
VOICE_WEIGHT = 0.55

RISK_THRESHOLDS = {
    "Low": (0, 30),
    "Moderate": (30, 55),
    "High": (55, 80),
    "Critical": (80, 101),
}

RECOMMENDED_ACTIONS = {
    "Low": ["Log interaction", "Standard follow-up"],
    "Moderate": ["Offer counselling referral", "Flag for follow-up call within 48h"],
    "High": [
        "Priority counselling referral",
        "Legal aid intervention",
        "Supervisor review within 24h",
    ],
    "Critical": [
        "Immediate human escalation",
        "Emergency support / crisis counsellor",
        "Consider police intervention or witness protection",
        "Do not close case without human sign-off",
    ],
}

# High-recall phrase list for suicidal ideation / extreme crisis language.
# This is intentionally broad (favors false positives over false negatives)
# since a missed critical case is far costlier than an unnecessary human
# review. Expand/localize this list per supported language.
_CRITICAL_PATTERNS = [
    r"\bkill (myself|me)\b",
    r"\bend (my|this) life\b",
    r"\bwant to die\b",
    r"\bno reason to live\b",
    r"\bbetter off dead\b",
    r"\bcan'?t (go on|take (it|this) anymore)\b",
    r"\bsuicid\w*\b",
    r"\bgoing to (hurt|kill) (myself|me)\b",
]
_CRITICAL_REGEX = re.compile("|".join(_CRITICAL_PATTERNS), re.IGNORECASE)


def detect_critical_flags(text):
    """
    Independent safety check on the raw transcript text.
    Returns (is_critical, matched_snippet_or_None).
    """
    if not text:
        return False, None

    match = _CRITICAL_REGEX.search(text)

    if match:
        return True, match.group(0)

    return False, None


def _text_distress_score(text_features):
    """
    Collapse the text_features dict into a single 0-1 text distress score.
    Prefers the ML-based emotion signal (ml_distress) when available, since
    it generalizes beyond exact keyword matches; falls back to the
    keyword-density weighting if the emotion model wasn't available.
    """
    if not text_features:
        return 0.0

    keyword_weighted = (
        0.30 * text_features.get("fear", 0.0)
        + 0.25 * text_features.get("distress", 0.0)
        + 0.20 * text_features.get("anxiety", 0.0)
        + 0.10 * text_features.get("uncertainty", 0.0)
        + 0.10 * text_features.get("sleep_distress", 0.0)
        + 0.05 * text_features.get("hesitation", 0.0)
    )
    keyword_weighted = min(keyword_weighted, 1.0)

    ml_distress = text_features.get("ml_distress")

    if ml_distress is not None:
        blended = 0.35 * keyword_weighted + 0.65 * ml_distress
    else:
        blended = keyword_weighted

    return float(min(blended, 1.0))


def _category_for_score(svi_score):
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= svi_score < high:
            return category

    return "Critical"


def compute_svi(text_features, voice_result, transcript=""):
    """
    Combine text + voice analysis into a final SVI result.

    text_features: dict from analysis.text_features.extract_text_features()
    voice_result: dict from analysis.voice_features.analyze_voice() (or None
                  if no audio buffer was available for this segment)
    transcript: the raw transcript text, used for the safety-override check
    """
    text_score = _text_distress_score(text_features)
    voice_score = voice_result["voice_score"] if voice_result else None

    if voice_score is not None:
        blended = TEXT_WEIGHT * text_score + VOICE_WEIGHT * voice_score
    else:
        # No usable audio for this segment -- fall back to text only
        blended = text_score

    svi_score = round(float(blended) * 100, 1)

    is_critical, critical_snippet = detect_critical_flags(transcript)

    if is_critical:
        # Safety override: force Critical regardless of blended score
        svi_score = max(svi_score, 90.0)
        category = "Critical"
    else:
        category = _category_for_score(svi_score)

    return {
        "svi_score": svi_score,
        "risk_category": category,
        "recommended_actions": RECOMMENDED_ACTIONS[category],
        "safety_override_triggered": is_critical,
        "safety_override_snippet": critical_snippet,
        "contributing_factors": {
            "text_score": round(text_score * 100, 1),
            "voice_score": round(voice_score * 100, 1) if voice_score is not None else None,
            "text_weight": TEXT_WEIGHT,
            "voice_weight": VOICE_WEIGHT if voice_score is not None else 0.0,
        },
    }
