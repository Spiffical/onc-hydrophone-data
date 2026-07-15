import io
from email.message import Message
import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_docs_figures.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_docs_figures",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
generate_docs_figures = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(generate_docs_figures)


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, content_type: str, status: int = 200):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_onc_audio_rejects_non_audio_response(tmp_path, monkeypatch):
    destination = tmp_path / "preview.mp3"
    monkeypatch.setattr(
        generate_docs_figures,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(b"<html>Access denied</html>", "text/html"),
    )

    with pytest.raises(RuntimeError, match="unexpected content type 'text/html'"):
        generate_docs_figures._download_onc_audio(destination)

    assert not destination.exists()
    assert not (tmp_path / "preview.mp3.part").exists()


def test_download_onc_audio_atomically_saves_audio(tmp_path, monkeypatch):
    destination = tmp_path / "preview.mp3"
    monkeypatch.setattr(
        generate_docs_figures,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(b"real audio bytes", "audio/mp3"),
    )

    generate_docs_figures._download_onc_audio(destination)

    assert destination.read_bytes() == b"real audio bytes"
    assert not (tmp_path / "preview.mp3.part").exists()
