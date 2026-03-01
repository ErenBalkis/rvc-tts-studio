"""
rvc_module.py – RVC inference logic for the TTS Studio.

Keeps all voice-conversion concerns out of the Streamlit UI layer.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
TEMP_DIR = Path(__file__).parent / "temp"

# Ensure directories exist at import time
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model Discovery
# ---------------------------------------------------------------------------
def list_models(models_dir: Optional[Path] = None) -> List[str]:
    """Return a sorted list of model names found in *models_dir*.

    Scans for subfolders that contain at least one ``.pth`` file.
    Each subfolder name is treated as a model name (e.g. ``trump``,
    ``senol_gunes``).

    Parameters
    ----------
    models_dir : Path, optional
        Directory to scan.  Defaults to ``./models``.

    Returns
    -------
    list[str]
        Names of subfolders containing a ``.pth`` file.
    """
    directory = Path(models_dir) if models_dir else MODELS_DIR
    if not directory.exists():
        logger.warning("Models directory does not exist: %s", directory)
        return []

    model_folders = sorted(
        d.name
        for d in directory.iterdir()
        if d.is_dir()
        and any(f.suffix.lower() == ".pth" for f in d.iterdir())
    )
    logger.info("Found %d model(s) in %s", len(model_folders), directory)
    return model_folders


def get_model_pth(model_name: str, models_dir: Optional[Path] = None) -> Optional[str]:
    """Return the full path to the ``.pth`` file inside a model subfolder.

    Parameters
    ----------
    model_name : str
        Subfolder name under *models_dir*.
    models_dir : Path, optional
        Parent directory. Defaults to ``./models``.

    Returns
    -------
    str or None
        Absolute path to the first ``.pth`` file found, or *None*.
    """
    directory = (Path(models_dir) if models_dir else MODELS_DIR) / model_name
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if f.suffix.lower() == ".pth":
            return str(f)
    return None


def find_index_file(model_name: str, models_dir: Optional[Path] = None) -> Optional[str]:
    """Try to locate a ``.index`` file inside the model's subfolder.

    Parameters
    ----------
    model_name : str
        Subfolder name under *models_dir*.
    models_dir : Path, optional
        Parent directory. Defaults to ``./models``.

    Returns
    -------
    str or None
        Full path to the index file, or *None* if not found.
    """
    directory = (Path(models_dir) if models_dir else MODELS_DIR) / model_name
    if not directory.exists():
        return None

    # Return first .index file found in the subfolder
    for f in directory.iterdir():
        if f.suffix.lower() == ".index":
            return str(f)

    return None


# ---------------------------------------------------------------------------
# Voice Conversion
# ---------------------------------------------------------------------------

# Supported f0 extraction methods (ordered by quality)
F0_METHODS = ["rmvpe", "harvest", "crepe", "pm"]


def convert_voice(
    model_path: str | Path,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    pitch_change: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
) -> str:
    """Run RVC inference to convert *input_audio_path* using *model_path*.

    This function bypasses ``RVCInference.infer_file`` and calls
    ``vc_single`` directly to work around a known tuple-return bug.

    Parameters
    ----------
    model_path : str | Path
        Full path to the ``.pth`` RVC model file.
    input_audio_path : str | Path
        Path to the source audio (e.g. the edge-tts MP3).
    output_audio_path : str | Path
        Where to write the converted WAV.
    pitch_change : int
        Semitone pitch shift for f0 extraction (–24 to +24).
    f0_method : str
        Pitch extraction algorithm: ``rmvpe`` (best quality),
        ``harvest``, ``crepe``, or ``pm``.
    index_rate : float
        Feature search ratio (0.0–1.0). Higher = more target voice character.
    filter_radius : int
        Median filtering radius for pitch (reduces breathiness).
    rms_mix_rate : float
        Volume envelope mix rate. Lower preserves more original dynamics.
    protect : float
        Protection for voiceless consonants (0.0–0.5).

    Returns
    -------
    str
        Absolute path to the output audio file on success.

    Raises
    ------
    FileNotFoundError
        If the model or input audio does not exist.
    RuntimeError
        If the RVC conversion fails for any reason.
    """
    model_path = Path(model_path)
    input_audio_path = Path(input_audio_path)
    output_audio_path = Path(output_audio_path)

    # --- Pre-flight checks ---------------------------------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"RVC model not found: {model_path}")
    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    # Ensure output directory exists
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Try to find a matching .index file -----------------------------------
    index_file = find_index_file(model_path.parent.name)

    # --- Run RVC conversion via rvc-python ------------------------------------
    try:
        from rvc_python.infer import RVCInference
        from scipy.io import wavfile
        import numpy as np

        device = "cuda:0" if _cuda_available() else "cpu:0"
        rvc = RVCInference(device=device)

        # Load model (pass index_path if available)
        rvc.load_model(
            str(model_path),
            version="v2",
            index_path=index_file or "",
        )

        logger.info(
            "Model loaded: %s | Index: %s | Device: %s",
            model_path.name,
            index_file or "none",
            device,
        )

        # --- Bypass infer_file — call vc_single directly ----
        # This avoids the tuple-return bug in some rvc-python versions.
        result = rvc.vc.vc_single(
            sid=0,
            input_audio_path=str(input_audio_path),
            f0_up_key=pitch_change,
            f0_method=f0_method,
            file_index=index_file or "",
            index_rate=index_rate if index_file else 0.0,
            filter_radius=filter_radius,
            resample_sr=0,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            f0_file="",
            file_index2="",
        )

        # vc_single may return (info_str, audio_np) or just audio_np
        if isinstance(result, tuple):
            wav_data = result[-1]  # last element is the audio array
            logger.info("vc_single info: %s", result[0] if len(result) > 1 else "")
        else:
            wav_data = result

        # Ensure correct dtype for scipy wavfile
        if hasattr(wav_data, "dtype") and wav_data.dtype != np.int16:
            # Normalise float audio to int16 range
            if np.issubdtype(wav_data.dtype, np.floating):
                peak = np.abs(wav_data).max()
                if peak > 0:
                    wav_data = (wav_data / peak * 32767).astype(np.int16)
                else:
                    wav_data = wav_data.astype(np.int16)

        wavfile.write(str(output_audio_path), rvc.vc.tgt_sr, wav_data)

        logger.info("Conversion complete → %s", output_audio_path)
        return str(output_audio_path.resolve())

    except ImportError as exc:
        raise RuntimeError(
            "rvc-python is not installed. "
            "Run:  pip install rvc-python>=0.1.5"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"RVC conversion failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    """Check whether CUDA is available without crashing if torch is missing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
