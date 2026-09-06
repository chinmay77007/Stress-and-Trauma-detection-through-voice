"""
Persistence layer: stores session metadata, per-utterance transcripts/SVI
scores, and the actual recorded audio (as WAV files on disk, referenced by
path from SQLite -- storing large binary blobs directly in the DB doesn't
scale well, plain files do).

Schema:
  sessions   -- one row per call/recording session
  utterances -- one row per finalized utterance within a session

NOTE ON PRIVACY: this stores real recorded speech from (potentially) SC/ST
atrocity victims describing violence, trauma, and personal details. Treat
this data directory and DB file as sensitive by default:
  - don't commit `recordings/` or `sessions.db` to git (already covered by
    a broad .gitignore pattern below -- double check it's actually there)
  - for anything beyond a local hackathon demo, this needs encryption at
    rest, access control, and a defined retention/deletion policy before
    it touches real victim data
"""

import os
import sqlite3
import uuid
import wave
from datetime import datetime, timezone

DB_PATH = "sessions.db"
RECORDINGS_DIR = "recordings"


def init_db():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            full_audio_path TEXT,
            peak_svi_score REAL,
            peak_risk_category TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text TEXT,
            svi_score REAL,
            risk_category TEXT,
            audio_path TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    conn.commit()
    conn.close()


def new_session_id():
    return uuid.uuid4().hex


def create_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
        (session_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def save_wav(pcm_bytes, filepath, sample_rate=16000, channels=1, sample_width=2):
    """Write raw Int16 PCM bytes out as a playable .wav file."""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def save_utterance(session_id, text, svi_score, risk_category, pcm_bytes, utterance_index):
    """
    Save one utterance's audio to disk and record it + its scores in the DB.
    Returns the audio file path (or None if there was no audio to save).
    """
    audio_path = None

    if pcm_bytes and len(pcm_bytes) > 0:
        session_dir = os.path.join(RECORDINGS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        audio_path = os.path.join(session_dir, f"utterance_{utterance_index:04d}.wav")
        save_wav(pcm_bytes, audio_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO utterances
           (session_id, timestamp, text, svi_score, risk_category, audio_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            datetime.now(timezone.utc).isoformat(),
            text,
            svi_score,
            risk_category,
            audio_path,
        ),
    )
    conn.commit()
    conn.close()

    return audio_path


def finalize_session(session_id, full_session_pcm_bytes, peak_svi):
    """
    Called when a call ends: saves the full concatenated session audio and
    closes out the session row with end time + peak risk reached.
    """
    full_audio_path = None

    if full_session_pcm_bytes and len(full_session_pcm_bytes) > 0:
        session_dir = os.path.join(RECORDINGS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        full_audio_path = os.path.join(session_dir, "full_session.wav")
        save_wav(bytes(full_session_pcm_bytes), full_audio_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE sessions
           SET ended_at = ?, full_audio_path = ?, peak_svi_score = ?, peak_risk_category = ?
           WHERE session_id = ?""",
        (
            datetime.now(timezone.utc).isoformat(),
            full_audio_path,
            peak_svi.get("svi_score"),
            peak_svi.get("risk_category"),
            session_id,
        ),
    )
    conn.commit()
    conn.close()

    return full_audio_path
