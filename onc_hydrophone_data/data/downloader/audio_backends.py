try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore
