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


@pytest.mark.parametrize(
    ("output_stem", "message"),
    [
        ("event.mat", "must not include a file extension"),
        ("event.", "must not include a file extension"),
        ("nested/event", "must be a filename stem"),
        (r"nested\\event", "must be a filename stem"),
    ],
)
def test_process_single_file_rejects_invalid_output_stems_before_loading(
    tmp_path: Path,
    output_stem: str,
    message: str,
):
    generator = SpectrogramGenerator(quiet=True)
    output_dir = tmp_path / "spectrograms"

    with pytest.raises(ValueError, match=message):
        generator.process_single_file(
            tmp_path / "missing.wav",
            output_dir,
            output_stem=output_stem,
        )

    assert not output_dir.exists()


def test_process_event_uses_complete_windows_and_trims_context(tmp_path: Path):
    sample_rate = 1_000
    duration_seconds = 20
    time_axis = np.arange(sample_rate * duration_seconds) / sample_rate
    signal = (
        np.sin(2 * np.pi * 80 * time_axis)
        + 0.4 * np.sin(2 * np.pi * 180 * time_axis)
    )
    # A much stronger signal outside the target interval verifies that extra
    # computation context is excluded before relative-dB normalization.
    signal[(time_axis >= 14.0) & (time_axis < 15.0)] *= 20.0
    audio_path = tmp_path / "events.wav"
    sf.write(audio_path, signal, sample_rate, subtype="FLOAT")

    generator = SpectrogramGenerator(
        win_dur=2.0,
        overlap=0.5,
        backend="scipy",
        quiet=True,
    )
    loaded_audio, loaded_rate, _ = generator.load_audio(audio_path)
    _, full_times, full_power, _ = generator.compute_spectrogram(
        loaded_audio,
        loaded_rate,
    )

    result = generator.process_event(
        audio_path,
        tmp_path / "spectrograms",
        event_time_seconds=10.0,
        pad_before_seconds=3.0,
        pad_after_seconds=3.0,
        save_plot=False,
        save_mat=False,
    )

    target_mask = (full_times >= 7.0) & (full_times <= 13.0)
    np.testing.assert_allclose(
        result["times"],
        full_times[target_mask] - 7.0,
    )
    np.testing.assert_allclose(
        result["power_spectrogram"],
        full_power[:, target_mask],
        rtol=1e-6,
        atol=1e-12,
    )
    assert result["duration"] == pytest.approx(6.0)
    assert result["edge_padding_seconds"] == pytest.approx(1.0)
    assert result["target_start_seconds"] == pytest.approx(7.0)
    assert result["target_end_seconds"] == pytest.approx(13.0)
    assert generator.clip_start is None
    assert generator.clip_end is None

    wider_context_result = generator.process_event(
        audio_path,
        tmp_path / "spectrograms",
        event_time_seconds=10.0,
        pad_before_seconds=3.0,
        pad_after_seconds=3.0,
        edge_padding_seconds=2.0,
        save_plot=False,
        save_mat=False,
    )
    assert np.max(wider_context_result["power_db_norm"]) == pytest.approx(0.0)

    default_result = generator.process_event(
        audio_path,
        tmp_path / "spectrograms",
        event_time_seconds=10.0,
        save_plot=False,
    )
    assert default_result["target_start_seconds"] == pytest.approx(5.0)
    assert default_result["target_end_seconds"] == pytest.approx(15.0)
    assert default_result["pad_before_seconds"] == pytest.approx(5.0)
    assert default_result["pad_after_seconds"] == pytest.approx(5.0)
    assert default_result["edge_padding_seconds"] == pytest.approx(1.0)
    assert Path(default_result["mat_file"]).name == "events_event_10000ms.mat"
    event_metadata = default_result["metadata"]["extra_metadata"]["event_spectrogram"]
    assert event_metadata["event_time_seconds"] == pytest.approx(10.0)
    assert event_metadata["edge_padding_requested"] == "auto"


@pytest.mark.parametrize(
    ("event_time", "pad_before", "pad_after", "message"),
    [
        (-1.0, 5.0, 5.0, "event_time_seconds"),
        (10.0, -1.0, 5.0, "padding values"),
        (10.0, 0.0, 0.0, "At least one"),
        (float("nan"), 5.0, 5.0, "finite"),
    ],
)
def test_process_event_validates_window(
    tmp_path: Path,
    event_time: float,
    pad_before: float,
    pad_after: float,
    message: str,
):
    generator = SpectrogramGenerator(quiet=True)

    with pytest.raises(ValueError, match=message):
        generator.process_event(
            tmp_path / "missing.wav",
            tmp_path,
            event_time_seconds=event_time,
            pad_before_seconds=pad_before,
            pad_after_seconds=pad_after,
        )


def test_process_event_rejects_negative_edge_padding_before_loading(tmp_path: Path):
    generator = SpectrogramGenerator(quiet=True)

    with pytest.raises(ValueError, match="edge_padding_seconds"):
        generator.process_event(
            tmp_path / "missing.wav",
            tmp_path,
            event_time_seconds=10.0,
            edge_padding_seconds=-0.1,
        )
