# app.py
import streamlit as st
import tempfile
import difflib
import json
import html
from pathlib import Path
from typing import Tuple, List
from openai import OpenAI
import time

# ---------- Helpers ----------
def safe_get_query_param(params, key, default=None):
    vals = params.get(key, [])
    return vals[0] if vals else default

def word_level_diff_html(target: str, spoken: str) -> Tuple[str, str, int]:
    """
    Return HTML for colored word-by-word diff for target and spoken.
    Correct words -> green, wrong/missing/extra -> red.
    Also returns matched word count for a basic accuracy calculation.
    """
    # Normalize and split by whitespace
    target_words = [w for w in target.strip().split() if w != ""]
    spoken_words = [w for w in spoken.strip().split() if w != ""]

    # Use SequenceMatcher on word lists
    sm = difflib.SequenceMatcher(a=target_words, b=spoken_words)
    # For building HTML
    target_html_parts = []
    spoken_html_parts = []
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for w in target_words[i1:i2]:
                target_html_parts.append(f'<span style="background:#d4f8d4;padding:2px;border-radius:4px;margin:1px;display:inline-block;">{html.escape(w)}</span>')
            for w in spoken_words[j1:j2]:
                spoken_html_parts.append(f'<span style="background:#d4f8d4;padding:2px;border-radius:4px;margin:1px;display:inline-block;">{html.escape(w)}</span>')
            matched += (i2 - i1)
        elif tag == "replace":
            # Words differ — red for both
            for w in target_words[i1:i2]:
                target_html_parts.append(f'<span style="background:#ffd6d6;padding:2px;border-radius:4px;margin:1px;display:inline-block;color:#b30000;">{html.escape(w)}</span>')
            for w in spoken_words[j1:j2]:
                spoken_html_parts.append(f'<span style="background:#ffd6d6;padding:2px;border-radius:4px;margin:1px;display:inline-block;color:#b30000;">{html.escape(w)}</span>')
        elif tag == "delete":
            # Present in target but missing in spoken
            for w in target_words[i1:i2]:
                target_html_parts.append(f'<span style="background:#ffd6d6;padding:2px;border-radius:4px;margin:1px;display:inline-block;color:#b30000;text-decoration:line-through;">{html.escape(w)}</span>')
        elif tag == "insert":
            # Extra in spoken
            for w in spoken_words[j1:j2]:
                spoken_html_parts.append(f'<span style="background:#ffd6d6;padding:2px;border-radius:4px;margin:1px;display:inline-block;color:#b30000;">{html.escape(w)}</span>')

    # Join
    target_html = " ".join(target_html_parts) if target_html_parts else "<i>(no words)</i>"
    spoken_html = " ".join(spoken_html_parts) if spoken_html_parts else "<i>(no words)</i>"

    # compute simple accuracy as matched / target_word_count
    target_count = max(1, len(target_words))
    accuracy_pct = int(round((matched / target_count) * 100))

    return target_html, spoken_html, accuracy_pct

def compute_char_similarity(a: str, b: str) -> int:
    """Alternative char-level similarity (0-100)."""
    sm = difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower())
    return int(round(sm.ratio() * 100))

# ---------- Streamlit UI & Logic ----------
st.set_page_config(page_title="English Pronunciation Coach", layout="centered")

st.title("🎧 English Pronunciation Coach")
st.write("Practice words or short sentences — record your voice and get friendly feedback.")

# Query params (hybrid behavior)
query_params = st.experimental_get_query_params()
url_word = safe_get_query_param(query_params, "word", None)
url_lang = safe_get_query_param(query_params, "lang", None)

# Defaults
default_text = url_word if url_word else "Hello"
selected_lang = url_lang if url_lang else "Amharic"

# Show small language notice and allow change if desired (but URL param preselects)
with st.sidebar:
    st.header("Settings")
    st.markdown("Language shown in feedback (from `?lang=` URL param). You can change it here too.")
    # Keep sidebar control so user can still switch language
    selected_lang = st.selectbox("Feedback Language", options=["Amharic", "Oromiffa", "English", "Amharic (Latin)"], index=0 if selected_lang=="Amharic" else (1 if selected_lang=="Oromiffa" else (2 if selected_lang=="English" else 0)))
    st.markdown("You can prefill the practice text with `?word=...` in the URL (e.g. `?word=Elephant`).")

# Show instructions depending on language
if selected_lang.lower().startswith("amhar"):
    st.info("ለመማር፡ ቃሉን አውትለኩ፣ መጀመሪያ 'Record' ይጫኑ፣ ከዚያ ድምጽዎን እና ማስተካከያን ያግኙ።")
elif selected_lang.lower().startswith("orom"):
    st.info("Barnootaaf: Jecha barreessi, 'Record' cuqaasi, sagalee kee galchi; deebii fi sirreessu argatta.")
else:
    st.info("Type the word or short sentence you want to practice, press Record, then submit — you'll get feedback in the selected language.")

# Text input prefilled by URL or default "Hello"
target_text = st.text_input("Target word / sentence to practice", value=default_text)

st.markdown("---")

# Audio input
st.subheader("Record your pronunciation")
st.write("Use the Record button below to capture your voice. (Streamlit's `st.audio_input`)")

audio_bytes = None
try:
    # Use st.audio_input (Streamlit 1.26+). If not available, fallback is handled by AttributeError.
    audio_bytes = st.audio_input("Press Record and speak", label_visibility="collapsed")
except Exception:
    # Fallback: file uploader (supports pre-recorded audio)
    st.warning("`st.audio_input` not available in this environment — please upload a short WAV/MP3 file instead.")
    uploaded = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded is not None:
        audio_bytes = uploaded.read()

# Button to submit / process
process = st.button("Process recording")

if process and audio_bytes:
    # Check for API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in `st.secrets['OPENAI_API_KEY']`. Please add it before continuing.")
    else:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        with st.spinner("Transcribing audio with Whisper & analyzing..."):
            # save audio_bytes to temp file — OpenAI expects a file-like object
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp_path = tmp.name
                # audio_bytes may be a bytes-like or a file-like, handle both
                if isinstance(audio_bytes, bytes):
                    tmp.write(audio_bytes)
                else:
                    # Streamlit's st.audio_input may return an UploadedFile-like object with getvalue()
                    try:
                        tmp.write(audio_bytes.getvalue())
                    except Exception:
                        # Last resort: try to read attribute 'read'
                        tmp.write(audio_bytes.read())
                tmp.flush()

            # Transcription via Whisper-1
            # NOTE: using OpenAI python package (OpenAI). Adjust if your environment uses different SDK.
            try:
                transcription_resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(tmp_path, "rb")
                )
                # Newer SDK returns object with "text"
                spoken_text = getattr(transcription_resp, "text", None) or transcription_resp.get("text") if isinstance(transcription_resp, dict) else str(transcription_resp)
                # If it's a dict and has 'text'
                if isinstance(spoken_text, str) and spoken_text.strip().lower().startswith("error"):
                    # fallback
                    spoken_text = ""
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                spoken_text = ""

            # Our simple local diff + accuracy
            target_html, spoken_html, word_accuracy = word_level_diff_html(target_text or "", spoken_text or "")
            char_accuracy = compute_char_similarity(target_text, spoken_text)

            st.markdown("### Results (local comparison)")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Target text**")
                st.markdown(target_html, unsafe_allow_html=True)
            with col2:
                st.markdown("**What you said**")
                st.markdown(spoken_html, unsafe_allow_html=True)

            st.markdown("")
            st.metric("Word-level accuracy (local)", f"{word_accuracy}%")
            st.metric("Character-level similarity (local)", f"{char_accuracy}%")

            # Now ask GPT-4o-mini to produce an analysis and list of mistakes in the chosen language.
            # We'll instruct it to respond with JSON only (for easier parsing). If that fails we'll show raw text.
            system_instruction = (
                f"You are a friendly, encouraging pronunciation coach. "
                f"Provide feedback and corrections in {selected_lang}. "
                f"Be concise, positive, and specific about pronunciation mistakes. "
                f"Return a JSON object only, with these keys:\n"
                f'  - "accuracy_percent" (number 0-100),\n'
                f'  - "mistakes" (array of short strings describing each mistake),\n'
                f'  - "detailed_feedback" (a short paragraph in {selected_lang} with tips),\n'
                f'  - "corrections" (optional: corrected phonetic hints or example words),\n'
                f'  - "display" (optional short human-friendly note to show on screen).\n'
                f'Do not include any other keys or explanation outside the JSON.'
            )

            user_message = (
                "Target text: " + target_text + "\n\n"
                "Spoken (transcription): " + spoken_text + "\n\n"
                "Please analyze pronunciation accuracy, list specific mistakes, and give correction tips."
            )

            try:
                chat_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=500,
                    temperature=0.2,
                )

                # Extract content
                gpt_text = ""
                if hasattr(chat_resp, "choices"):
                    # Newer shapes: choices[0].message.content
                    first = chat_resp.choices[0]
                    # many SDKs store message differently
                    try:
                        gpt_text = first.message.get("content") if getattr(first, "message", None) else str(first)
                    except Exception:
                        # fallback
                        gpt_text = str(first)
                else:
                    # fallback to raw
                    gpt_text = str(chat_resp)

            except Exception as e:
                st.error(f"AI analysis failed: {e}")
                gpt_text = ""

            # Try to parse JSON from GPT
            parsed = None
            if gpt_text:
                # Try to find a JSON substring
                try:
                    # gpt may include code fences or surrounding text. try to find first '{' and last '}'.
                    start = gpt_text.find("{")
                    end = gpt_text.rfind("}")
                    if start != -1 and end != -1:
                        json_str = gpt_text[start:end+1]
                        parsed = json.loads(json_str)
                    else:
                        parsed = json.loads(gpt_text)  # try direct load
                except Exception:
                    parsed = None

            st.markdown("---")
            st.header("AI Pronunciation Feedback")

            if parsed:
                # Show AI-provided accuracy if present
                ai_accuracy = parsed.get("accuracy_percent", None)
                if ai_accuracy is not None:
                    st.metric("AI accuracy estimate", f"{ai_accuracy}%")
                else:
                    st.info("AI did not return a numeric accuracy. Showing local metrics above.")

                # Mistakes list
                mistakes = parsed.get("mistakes", [])
                if mistakes:
                    st.subheader("Specific mistakes identified")
                    for m in mistakes:
                        st.write(f"- {m}")
                else:
                    st.write("_No mistakes were listed by the AI._")

                # Detailed feedback (in chosen language)
                detailed = parsed.get("detailed_feedback", "")
                if detailed:
                    st.subheader("Feedback")
                    st.write(detailed)

                # Corrections (phonetic hints or examples)
                corrections = parsed.get("corrections", "")
                if corrections:
                    st.subheader("Corrections / Examples")
                    st.write(corrections)

                # Optional display text
                display_note = parsed.get("display", "")
                if display_note:
                    st.info(display_note)

            else:
                st.warning("Could not parse structured feedback from the AI. Raw AI response follows.")
                st.write("**Raw AI response:**")
                st.write(gpt_text)

            st.markdown("---")
            st.caption("Local comparison (green = matched, red = mismatched). AI feedback is intended to be friendly and corrective.")
            # small telemetry/time stamp
            st.write(f"_Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}_")

        # End of processing block

elif process and not audio_bytes:
    st.error("No audio captured. Make sure you pressed Record and allowed microphone access, or upload an audio file in the fallback uploader.")

# Footer/help
st.markdown("---")
st.write("Built for pronunciation practice. Use short phrases or single words for best results.")
st.write("If you want the app to prefill a word, add `?word=Elephant` to the URL. To set the feedback language, add `?lang=Oromiffa` (default: Amharic).")
