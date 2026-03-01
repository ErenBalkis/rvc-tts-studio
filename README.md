# 🎙️ RVC Text-to-Speech Studio

Convert any text into a **custom voice** using [edge-tts](https://github.com/rany2/edge-tts) and [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

Built with **Streamlit** for a sleek, interactive web interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🗣️ **13+ Base Voices** — Choose from Turkish, English, German, French, Spanish, Japanese, Korean, and Chinese neural voices via edge-tts
- 🎤 **RVC Voice Conversion** — Transform the base speech into any custom voice with a trained `.pth` model
- 🎼 **Multiple F0 Methods** — `rmvpe` (best quality), `harvest`, `crepe`, `pm` (fastest)
- 🎵 **Pitch Control** — Adjust pitch ±24 semitones to match the target model's range
- ⚙️ **Fine-Tuning** — Index rate, RMS mix rate, consonant protection sliders
- 🖥️ **GPU & CPU Support** — Automatically uses CUDA if available, falls back to CPU
- ⬇️ **Download** — Listen in-browser and download the final WAV file

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ErenBalkis/rvc-tts-studio.git
cd rvc-tts-studio
```

### 2. Create a virtual environment (Python 3.10 recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch

> **⚠️ You must install the correct PyTorch build _before_ installing the other dependencies.**

| Hardware       | Command |
|----------------|---------|
| **CPU only**   | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| **CUDA 11.8**  | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| **CUDA 12.1**  | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your RVC models

Create a subfolder for each voice inside the `models/` directory. Each subfolder should contain a `.pth` file and optionally a `.index` file:

```
models/
├── my_voice/
│   ├── my_voice.pth
│   └── my_voice.index   ← optional, improves quality
├── another_voice/
│   └── another_voice.pth
└── .gitkeep
```

> 💡 **Where to find RVC models?** Search for pre-trained RVC models on [Hugging Face](https://huggingface.co/) or [weights.gg](https://weights.gg/).

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
rvc-tts-studio/
├── models/            # Your RVC model subfolders (.pth & .index)
│   └── .gitkeep       # Keeps the empty folder in git
├── temp/              # Temporary audio files (auto-managed)
│   └── .gitkeep
├── app.py             # Streamlit UI & application logic
├── rvc_module.py      # RVC inference & model discovery module
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

---

## 🎯 Usage

1. **Select a model** from the sidebar dropdown (click *🔄 Refresh Models* after adding new files).
2. **Type or paste** your text in the main area.
3. **Choose a base voice** — the edge-tts speaker used before RVC conversion.
4. **Adjust the pitch slider** to match your target model's vocal range (positive = higher, negative = lower).
5. **Fine-tune quality settings** in the sidebar (F0 method, index rate, RMS mix, protect).
6. Click **🚀 Generate Voice** and wait for the result.
7. **Listen** in-browser or **⬇️ Download** the final WAV.

---

## ⚙️ Quality Settings

| Setting | Description | Recommended |
|---------|-------------|-------------|
| **F0 Method** | Pitch extraction algorithm | `rmvpe` for best quality |
| **Index Rate** | How much target voice character to apply (0.0–1.0) | `0.75` |
| **RMS Mix Rate** | Volume envelope mixing (0.0–1.0) | `0.25` |
| **Protect** | Consonant protection against artifacts (0.0–0.5) | `0.33` |
| **Pitch Shift** | Semitone adjustment (-24 to +24) | `0` (adjust per model) |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `numpy` version conflict | Make sure `numpy < 2.0` is installed: `pip install "numpy<2.0"` |
| `torch` not found | Install PyTorch manually using the table in Step 3 above |
| No models in dropdown | Place `.pth` files inside subfolders in `models/` and click **Refresh Models** |
| CUDA out of memory | Use a smaller model or switch to CPU (`torch` CPU build) |
| `rvc-python` import error | Run `pip install rvc-python>=0.1.5` |

---

## 🛡️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — Interactive web UI
- **[edge-tts](https://github.com/rany2/edge-tts)** — Microsoft Edge's online TTS engine
- **[rvc-python](https://pypi.org/project/rvc-python/)** — RVC inference wrapper
- **[PyTorch](https://pytorch.org/)** — Deep learning backend
- **[librosa](https://librosa.org/)** — Audio processing
- **[NumPy](https://numpy.org/)** / **[SciPy](https://scipy.org/)** — Numerical computing

---