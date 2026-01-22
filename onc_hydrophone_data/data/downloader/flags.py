from typing import Optional


def _resolve_download_audio(
    download_audio: Optional[bool],
    download_flac: Optional[bool],
) -> bool:
    """Resolve download_audio/download_flac alias flags into a single boolean."""
    if download_audio is None and download_flac is None:
        return False
    if download_audio is None:
        return bool(download_flac)
    if download_flac is not None and download_flac != download_audio:
        raise ValueError("download_audio and download_flac provide conflicting values")
    return bool(download_audio)
