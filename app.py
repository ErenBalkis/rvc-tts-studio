"""
app.py – RVC Text-to-Speech Studio

Streamlit interface that:
  1. Takes user text input
  2. Synthesises base audio with edge-tts
  3. Converts the audio to the target voice via an RVC model
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rvc_module import list_models, get_model_pth, convert_voice, MODELS_DIR, TEMP_DIR, F0_METHODS  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RVC Text-to-Speech Studio",
    page_icon="🎙️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* ---- Header ---- */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #aaa;
        margin-bottom: 2rem;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #1a1a2e !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #00d2ff;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(0, 210, 255, 0.35);
    }

    /* ---- Audio player ---- */
    .stAudio {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Available edge-tts voices (Turkish defaults first, then common ones)
# ---------------------------------------------------------------------------
EDGE_TTS_VOICES = [
    "tr-TR-AhmetNeural",
    "tr-TR-EmelNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "de-DE-ConradNeural",
    "fr-FR-HenriNeural",
    "es-ES-AlvaroNeural",
    "ja-JP-KeitaNeural",
    "ko-KR-InJoonNeural",
    "zh-CN-YunxiNeural",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_edge_tts(text: str, voice: str, output_path: str) -> None:
    """Synthesise *text* with edge-tts and save to *output_path*."""
    import edge_tts

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    # Use a fresh event loop to avoid conflicts with Streamlit's loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_generate())
    finally:
        loop.close()


def _refresh_models() -> list[str]:
    """Re-scan the models directory and update session state."""
    models = list_models()
    st.session_state["models"] = models
    return models


# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------
if "models" not in st.session_state:
    _refresh_models()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Model Settings")

    models: list[str] = st.session_state.get("models", [])

    if st.button("🔄 Refresh Models", use_container_width=True):
        models = _refresh_models()
        if models:
            st.success(f"Found **{len(models)}** model(s).")
        else:
            st.warning("No model folders found in `models/`.")

    if models:
        selected_model = st.selectbox(
            "Select RVC Model",
            options=models,
            help="Each subfolder in `models/` should contain a `.pth` and optionally a `.index` file.",
        )
    else:
        st.info("📂 Create a subfolder in **models/** with a `.pth` file inside and click **Refresh Models**.")
        selected_model = None

    st.markdown("---")
    st.markdown("## ⚙️ Quality Settings")

    f0_method = st.selectbox(
        "🎼 F0 Method (pitch extraction)",
        options=F0_METHODS,
        index=0,
        help="**rmvpe** = best quality (slower). **pm** = fastest (lower quality).",
    )

    index_rate = st.slider(
        "🔍 Index Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher = more target voice character. Only effective if .index file exists.",
    )

    rms_mix_rate = st.slider(
        "🔊 RMS Mix Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = preserves original dynamics. Higher = more uniform volume.",
    )

    protect = st.slider(
        "🛡️ Protect (consonants)",
        min_value=0.0,
        max_value=0.5,
        value=0.33,
        step=0.01,
        help="Protects voiceless consonants from artifacts. Higher = more protection.",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#888;">
        Built with ❤️ using<br>
        <b>Streamlit</b> · <b>edge-tts</b> · <b>RVC</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🎙️ RVC Text-to-Speech Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Type your script, choose a voice, and let RVC transform it.</p>',
    unsafe_allow_html=True,
)

# --- Text input ---
text_input = st.text_area(
    "📝 Enter your text",
    height=180,
    placeholder="Merhaba! Bu bir ses dönüştürme testidir...",
)

# --- Voice & Pitch controls ---
col1, col2 = st.columns([3, 2])

with col1:
    base_voice = st.selectbox(
        "🗣️ Base Voice (edge-tts)",
        options=EDGE_TTS_VOICES,
        index=0,
        help="The initial TTS voice. The RVC model will transform it afterwards.",
    )

with col2:
    pitch_shift = st.slider(
        "🎵 Pitch Shift (semitones)",
        min_value=-24,
        max_value=24,
        value=0,
        help="Positive → higher, Negative → lower. Adjust to match the target model's range.",
    )

st.markdown("---")

# --- Generate button ---
generate_clicked = st.button("🚀 Generate Voice", use_container_width=True, type="primary")

if generate_clicked:
    # Validation
    if not text_input.strip():
        st.error("⚠️ Please enter some text before generating.")
        st.stop()
    if selected_model is None:
        st.error("⚠️ No RVC model selected. Add a model folder to `models/` and click **Refresh Models**.")
        st.stop()

    base_audio_path = str(TEMP_DIR / "base_tts.mp3")
    output_audio_path = str(TEMP_DIR / "output.wav")
    model_full_path = get_model_pth(selected_model)
    if model_full_path is None:
        st.error(f"⚠️ No `.pth` file found in `models/{selected_model}/`.")
        st.stop()

    # Step 1 – edge-tts
    with st.status("🔊 Generating base speech with edge-tts...", expanded=True) as status:
        try:
            _run_edge_tts(text_input, base_voice, base_audio_path)
            status.update(label="✅ Base speech generated!", state="complete")
        except Exception as exc:
            st.error(f"❌ edge-tts failed: {exc}")
            st.stop()

    # Step 2 – RVC conversion
    with st.status("🎤 Converting voice with RVC model...", expanded=True) as status:
        try:
            result_path = convert_voice(
                model_path=model_full_path,
                input_audio_path=base_audio_path,
                output_audio_path=output_audio_path,
                pitch_change=pitch_shift,
                f0_method=f0_method,
                index_rate=index_rate,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )
            status.update(label="✅ Voice conversion complete!", state="complete")
        except FileNotFoundError as exc:
            st.error(f"❌ File not found: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"❌ RVC error: {exc}")
            st.stop()

    # Step 3 – Playback & Download
    st.markdown("### 🎧 Result")
    with open(result_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label="⬇️ Download WAV",
            data=audio_bytes,
            file_name="rvc_output.wav",
            mime="audio/wav",
            use_container_width=True,
        )
