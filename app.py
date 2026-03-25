from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
@dataclass(frozen=True)
class AppConfig:
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int

    groq_api_key: str
    groq_model_id: str
    system_prompt: str

    tts_engine: str
    gtts_lang: str
    piper_model_path: str
    piper_config_path: str


def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except:
            return default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en"),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),

        groq_api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant"),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant for students. Keep replies short and clear."
        ),

        tts_engine=os.getenv("TTS_ENGINE", "gtts"),
        gtts_lang=os.getenv("GTTS_LANG", "en"),
        piper_model_path=os.getenv("PIPER_MODEL_PATH", ""),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", ""),
    )

CFG = load_config()

# ================= ASR =================
@st.cache_resource
def get_whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel(
        CFG.whisper_model,
        device=CFG.whisper_device,
        compute_type=CFG.whisper_compute_type,
        cpu_threads=CFG.whisper_cpu_threads,
    )


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        segments, _ = model.transcribe(
            tmp_path,
            language=CFG.whisper_language,
            beam_size=1,
            vad_filter=True
        )
        return "".join(seg.text for seg in segments).strip()
    finally:
        os.remove(tmp_path)

# ================= LLM =================
def offline_demo_reply(user_text: str) -> str:
    return f"Offline mode.\nYou said: {user_text}"


def groq_chat_completion(messages: List[Dict[str, str]]) -> str:
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CFG.groq_api_key}"}

    payload = {
        "model": CFG.groq_model_id,
        "messages": messages,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def generate_reply(user_text: str, history: List[Dict[str, str]]) -> str:
    if not CFG.groq_api_key:
        return offline_demo_reply(user_text)

    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    return groq_chat_completion(messages)

# ================= TTS =================
def tts_to_audio_file(text: str, voice: str = None) -> Tuple[bytes, str, str]:
    if CFG.tts_engine == "piper":
        return piper_tts(text, voice)
    return gtts_tts(text)


def gtts_tts(text: str):
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name

    tts = gTTS(text=text, lang=CFG.gtts_lang)
    tts.save(path)

    with open(path, "rb") as f:
        audio = f.read()

    os.remove(path)
    return audio, "audio/mpeg", "reply.mp3"


def piper_tts(text: str, voice_name: str = None):
    try:
        from piper import PiperVoice
        import wave

        if not CFG.piper_model_path or not CFG.piper_config_path:
            return gtts_tts("Piper not configured properly.")

        voice = PiperVoice.load(CFG.piper_model_path, CFG.piper_config_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        with wave.open(path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        with open(path, "rb") as f:
            audio = f.read()

        os.remove(path)
        return audio, "audio/wav", "reply.wav"

    except Exception:
        return gtts_tts("Piper failed, fallback to gtts.")

# ================= Streamlit UI =================

st.set_page_config(page_title="Speech-to-Speech", layout="centered")
st.title("🎙️ VoiceBridge Speech to Speech AI")

with st.sidebar:
    st.subheader("Settings")

    # Language Switch (NEW)
    language = st.selectbox("Language", ["en", "ur"])
    CFG.whisper_language = language
    CFG.gtts_lang = language

    # Engine switch
    voice_option = st.selectbox("TTS Engine", ["gtts", "piper"])
    CFG.tts_engine = voice_option

    # Piper voice dropdown (NEW)
    piper_voices = ["en_US-lessac", "en_US-ryan", "en_GB-alan"]
    selected_voice = st.selectbox("Piper Voice", piper_voices)

    # Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

st.write("### 🎤 Record your voice")

audio_value = st.audio_input("")

if audio_value:
    wav_bytes = audio_value.getvalue()

    if not wav_bytes:
        st.warning("⚠️ No audio detected!")
    else:
        st.audio(audio_value)

        with st.spinner("Processing..."):

            start = time.time()
            transcript = transcribe_wav_bytes(wav_bytes)
            asr_time = time.time() - start

            # Noise handling (NEW)
            if not transcript.strip() or len(transcript.split()) < 2:
                st.warning("⚠️ Audio too short or noisy. Try again.")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": transcript}
                )

                with st.chat_message("user"):
                    st.write(transcript)

                start = time.time()
                reply_text = generate_reply(
                    transcript, st.session_state.chat_history
                )
                llm_time = time.time() - start

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply_text}
                )

                with st.chat_message("assistant"):
                    st.write(reply_text)

                start = time.time()
                audio_bytes, mime, fname = tts_to_audio_file(
                    reply_text, selected_voice
                )
                tts_time = time.time() - start

                st.audio(audio_bytes, format=mime)

                st.download_button(
                    "⬇️ Download reply audio",
                    data=audio_bytes,
                    file_name=fname,
                    mime=mime
                )

                st.caption(
                    f"⏱️ ASR: {asr_time:.2f}s | LLM: {llm_time:.2f}s | TTS: {tts_time:.2f}s"
                )

st.divider()

st.write("## Debug / Helper")

if st.checkbox("Show chat history"):
    st.json(st.session_state.chat_history)
