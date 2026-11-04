"""Qirat Tajweed Checker - Streamlit + Gemini Flash.

Upload (or record) a Quran recitation, and the AI gives a Tajweed score,
lists mistakes with timestamps, and suggests pronunciation corrections.
"""

import html
import json
import os
import re
import time

import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - shown in UI if package is missing
    genai = None
    types = None

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_AUDIO_MB = 15  # inline upload limit stays safely under Gemini's request cap

MIME_TYPES = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "aiff": "audio/aiff",
    "m4a": "audio/mp4",
}

# Category -> weight in the overall score (weights sum to 1.0)
CATEGORIES = {
    "makharij": ("Makharij (articulation points)", 0.25),
    "sifat": ("Sifat (letter characteristics)", 0.15),
    "madd": ("Madd (elongation)", 0.20),
    "ghunnah_noon_meem": ("Ghunnah, Noon Sakinah & Meem Sakinah", 0.20),
    "qalqalah": ("Qalqalah", 0.05),
    "waqf_ibtida": ("Waqf & Ibtida (stopping/starting)", 0.05),
    "fluency": ("Fluency & steadiness (Tartil)", 0.10),
}

SYSTEM_PROMPT = """You are an experienced Quran recitation (Tajweed) teacher.
You will listen to an audio recording of someone reciting the Quran and
evaluate it strictly and fairly according to the rules of Tajweed.

Rules for your evaluation:
- Listen carefully to the audio. Only report mistakes you can actually hear.
  Never invent mistakes to seem thorough, and never invent praise.
- If the audio is not Quran recitation, is silent, or is too noisy/short to
  judge, set "audio_usable" to false, explain in "audio_note", and give
  null for all scores.
- Judge against the requested riwayah/qirat (default: Hafs 'an 'Asim).
- If a reference text is provided, compare the recitation to it and mention
  skipped, repeated or changed words. If not, identify the verses yourself
  and state your confidence honestly.
- Give timestamps (mm:ss) for each mistake where possible.
- Write corrections in simple English, give the Arabic word, and describe
  HOW to fix the pronunciation (tongue position, duration in counts, etc.).
- Scores are integers from 0 to 100 for each category.
- Respond with JSON only, following the schema exactly."""

SCHEMA_HINT = {
    "audio_usable": True,
    "audio_note": "string (why unusable, or short note on audio quality)",
    "detected_text": "Arabic text or verse reference you heard (may be empty)",
    "detection_confidence": "low | medium | high",
    "scores": {key: "integer 0-100" for key in CATEGORIES},
    "mistakes": [
        {
            "timestamp": "mm:ss",
            "word": "Arabic word or phrase",
            "rule": "Tajweed rule involved",
            "issue": "what was heard",
            "correction": "how to fix it",
            "severity": "minor | moderate | major",
        }
    ],
    "strengths": ["string"],
    "practice_tips": ["string"],
    "summary": "2-3 sentence overall feedback",
}


# ----------------------------------------------------------------------------
# Pure helper functions (no Streamlit / network needed, easy to test)
# ----------------------------------------------------------------------------
def get_mime_type(filename: str, fallback: str | None = None) -> str | None:
    """Return the Gemini-compatible MIME type for an audio filename."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return MIME_TYPES.get(ext, fallback)


def build_user_prompt(riwayah: str, strictness: str, reference_text: str) -> str:
    parts = [
        f"Riwayah/Qirat: {riwayah}.",
        f"Strictness: {strictness}.",
    ]
    if reference_text.strip():
        parts.append(f"Reference text the reciter intends to read:\n{reference_text.strip()}")
    else:
        parts.append("No reference text was provided; identify the verses yourself.")
    parts.append("Return JSON only in this shape:\n" + json.dumps(SCHEMA_HINT, ensure_ascii=False, indent=2))
    return "\n\n".join(parts)


def extract_json(text: str) -> dict:
    """Parse JSON from a model response, tolerating ```json fences and extra text."""
    if not text or not text.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model did not return valid JSON.")
        return json.loads(cleaned[start : end + 1])


def _clamp_score(value) -> int | None:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def normalize_result(raw: dict) -> dict:
    """Validate the model output and compute the weighted overall score."""
    usable = bool(raw.get("audio_usable", True))
    scores_in = raw.get("scores") or {}
    scores = {key: _clamp_score(scores_in.get(key)) for key in CATEGORIES}

    # Weighted average over the categories that actually have a score
    weight_sum = sum(CATEGORIES[k][1] for k, v in scores.items() if v is not None)
    if usable and weight_sum > 0:
        overall = round(sum(v * CATEGORIES[k][1] for k, v in scores.items() if v is not None) / weight_sum)
    else:
        overall = None
        usable = usable and weight_sum > 0

    mistakes = []
    for m in raw.get("mistakes") or []:
        if isinstance(m, dict):
            mistakes.append({
                "timestamp": str(m.get("timestamp", "")),
                "word": str(m.get("word", "")),
                "rule": str(m.get("rule", "")),
                "issue": str(m.get("issue", "")),
                "correction": str(m.get("correction", "")),
                "severity": str(m.get("severity", "minor")).lower(),
            })

    def as_list(x):
        return [str(i) for i in x] if isinstance(x, list) else []

    return {
        "audio_usable": usable,
        "audio_note": str(raw.get("audio_note", "") or ""),
        "detected_text": str(raw.get("detected_text", "") or ""),
        "detection_confidence": str(raw.get("detection_confidence", "") or ""),
        "scores": scores,
        "overall": overall,
        "mistakes": mistakes,
        "strengths": as_list(raw.get("strengths")),
        "practice_tips": as_list(raw.get("practice_tips")),
        "summary": str(raw.get("summary", "") or ""),
    }


def grade_label(score: int | None) -> str:
    if score is None:
        return "Not available"
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Needs practice"
    return "Needs a lot of practice"


# ----------------------------------------------------------------------------
# Gemini call
# ----------------------------------------------------------------------------
def analyze_recitation(api_key, model, audio_bytes, mime_type, riwayah, strictness, reference_text) -> dict:
    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Run: pip install -r requirements.txt")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
    )
    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        build_user_prompt(riwayah, strictness, reference_text),
    ]

    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return normalize_result(extract_json(response.text))
        except ValueError as exc:  # bad / empty JSON -> retry
            last_error = exc
        except Exception as exc:  # network, quota, auth ...
            message = str(exc)
            if any(code in message for code in ("API key", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT")):
                raise
            last_error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Analysis failed after 3 attempts: {last_error}")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # no secrets file locally
    return os.environ.get("GEMINI_API_KEY", "")


def show_results(result: dict) -> None:
    if not result["audio_usable"]:
        st.warning(result["audio_note"] or "The audio could not be evaluated. Try a clearer, longer recording.")
        return

    st.subheader("Your Tajweed score")
    col1, col2 = st.columns([1, 2])
    col1.metric("Overall", f"{result['overall']}/100")
    col2.markdown(f"### {grade_label(result['overall'])}")
    col2.progress(result["overall"] / 100)

    if result["summary"]:
        st.info(result["summary"])

    if result["detected_text"]:
        st.markdown("**Detected recitation**")
        st.markdown(f"<div dir='rtl' style='font-size:1.6rem'>{html.escape(result['detected_text'])}</div>", unsafe_allow_html=True)
        if result["detection_confidence"]:
            st.caption(f"Detection confidence: {result['detection_confidence']}")

    st.markdown("#### Score by category")
    for key, (label, _) in CATEGORIES.items():
        value = result["scores"].get(key)
        if value is None:
            st.write(f"{label}: n/a")
        else:
            st.write(f"{label}: **{value}**")
            st.progress(value / 100)

    st.markdown("#### Mistakes & corrections")
    if not result["mistakes"]:
        st.success("No clear mistakes were detected. MashaAllah!")
    icons = {"major": "🔴", "moderate": "🟠", "minor": "🟡"}
    for m in result["mistakes"]:
        title = f"{icons.get(m['severity'], '🟡')} {m['timestamp'] or '--:--'} - {m['word'] or 'Unknown word'} ({m['rule']})"
        with st.expander(title):
            st.markdown(f"**What was heard:** {m['issue']}")
            st.markdown(f"**How to correct it:** {m['correction']}")

    if result["strengths"]:
        st.markdown("#### Strengths")
        for s in result["strengths"]:
            st.write(f"- {s}")
    if result["practice_tips"]:
        st.markdown("#### Practice tips")
        for t in result["practice_tips"]:
            st.write(f"- {t}")

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="tajweed_report.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="Qirat Tajweed Checker", page_icon="📖", layout="centered")
    st.title("📖 Qirat Tajweed Checker")
    st.caption("Upload or record your Quran recitation. AI gives a Tajweed score and pronunciation corrections.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password", help="Get a free key at aistudio.google.com")
        else:
            st.success("API key loaded")
        model = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
        riwayah = st.selectbox("Riwayah / Qirat", ["Hafs 'an 'Asim", "Warsh 'an Nafi'", "Qalun 'an Nafi'", "Al-Duri 'an Abu 'Amr", "Shu'bah 'an 'Asim"])
        strictness = st.select_slider("Strictness", options=["Beginner-friendly", "Standard", "Strict (teacher level)"], value="Standard")

    st.warning(
        "AI feedback is a practice aid and can make mistakes. It is not a replacement "
        "for a qualified Tajweed teacher (with ijazah). Verify important corrections with one."
    )

    reference_text = st.text_area(
        "Verse text (optional)", placeholder="Paste the Arabic verses you are reciting for more accurate checking", height=100
    )

    tab_upload, tab_record = st.tabs(["Upload audio", "Record audio"])
    audio_bytes, mime_type, source_name = None, None, None

    with tab_upload:
        uploaded = st.file_uploader("Upload your recitation", type=list(MIME_TYPES.keys()))
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
            mime_type = get_mime_type(uploaded.name, uploaded.type)
            source_name = uploaded.name
            st.audio(audio_bytes)

    with tab_record:
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Record your recitation")
            if recorded is not None:
                audio_bytes = recorded.getvalue()
                mime_type = "audio/wav"
                source_name = "recording.wav"
        else:
            st.info("Recording needs Streamlit 1.39 or newer. Please upload a file instead.")

    if st.button("Analyze my recitation", type="primary", disabled=audio_bytes is None):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        elif not mime_type:
            st.error("Unsupported audio format. Please upload mp3, wav, ogg, flac, aac, aiff or m4a.")
        elif len(audio_bytes) > MAX_AUDIO_MB * 1024 * 1024:
            st.error(f"File is too large. Please keep it under {MAX_AUDIO_MB} MB (about 10-15 minutes of mp3).")
        elif len(audio_bytes) < 2000:
            st.error("The audio is too short. Please record a longer recitation.")
        else:
            with st.spinner(f"Listening to {source_name} ..."):
                try:
                    st.session_state["result"] = analyze_recitation(
                        api_key, model, audio_bytes, mime_type, riwayah, strictness, reference_text
                    )
                except Exception as exc:
                    st.session_state.pop("result", None)
                    st.error(f"Something went wrong: {exc}")

    if "result" in st.session_state:
        show_results(st.session_state["result"])


if __name__ == "__main__":
    main()
