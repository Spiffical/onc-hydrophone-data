from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from onc_hydrophone_data.audio.spectrogram_generator import (
    HAS_TORCH,
    SpectrogramGenerator,
)


@pytest.mark.skipif(not HAS_TORCH, reason="Torch/torchaudio are not installed")
def test_torch_and_scipy_psd_outputs_align():
    sample_rate = 8_000
    rng = np.random.default_rng(42)
    time_axis = np.arange(sample_rate * 3) / sample_rate
    audio = (
        0.6 * np.sin(2 * np.pi * 440 * time_axis)
        + 0.2 * np.sin(2 * np.pi * 1_200 * time_axis)
        + 0.01 * rng.standard_normal(time_axis.shape)
    )

    scipy_generator = SpectrogramGenerator(
        win_dur=0.25,
        overlap=0.5,
        backend="scipy",
        quiet=True,
    )
    torch_generator = SpectrogramGenerator(
        win_dur=0.25,
        overlap=0.5,
        backend="torch",
        torch_device="cpu",
        quiet=True,
    )

    scipy_f, scipy_t, scipy_power, scipy_db = scipy_generator.compute_spectrogram(
        audio, sample_rate
    )
    torch_f, torch_t, torch_power, torch_db = torch_generator.compute_spectrogram(
        audio, sample_rate
    )

    np.testing.assert_allclose(torch_f, scipy_f)
    np.testing.assert_allclose(torch_t, scipy_t)
    np.testing.assert_allclose(torch_power, scipy_power, rtol=2e-3, atol=1e-10)
    np.testing.assert_allclose(torch_db, scipy_db, rtol=2e-3, atol=2e-2)


def test_batch_quiet_mode_releases_arrays_and_preserves_order(tmp_path: Path, capsys):
    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "spectrograms"
    input_dir.mkdir()
    sample_rate = 4_000
    signal = np.sin(2 * np.pi * 300 * np.arange(sample_rate) / sample_rate)
    sf.write(input_dir / "b.wav", signal, sample_rate)
    sf.write(input_dir / "a.wav", signal, sample_rate)

    generator = SpectrogramGenerator(
        win_dur=0.1,
        backend="scipy",
        quiet=True,
    )
    results = generator.process_directory(
        input_dir,
        output_dir,
        save_plot=False,
        save_mat=False,
        max_workers=2,
    )

    assert [Path(result["audio_file"]).name for result in results] == ["a.wav", "b.wav"]
    for result in results:
        assert "power_spectrogram" not in result
        assert "power_db_norm" not in result
        assert "frequencies" not in result
        assert "times" not in result
    assert capsys.readouterr().out == ""


def test_hashable_windows_are_cached():
    generator = SpectrogramGenerator(window_type=("kaiser", 12.345), quiet=True)
    first = generator._resolve_window(321)
    second = generator._resolve_window(321)
    assert first is second
