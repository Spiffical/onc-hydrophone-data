import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ...onc.common import ensure_timezone_aware
from .models import TimestampRequest


def _resolve_custom_clip_pad_seconds(
    clip_pad_seconds: Union[float, str, None],
    generator_options: Dict[str, Any],
) -> float:
    if clip_pad_seconds is None or (
        isinstance(clip_pad_seconds, str) and clip_pad_seconds.strip().lower() == 'auto'
    ):
        win_dur = generator_options.get('win_dur', 1.0)
        overlap = generator_options.get('overlap', 0.5)
        try:
            win_dur = float(win_dur)
        except (TypeError, ValueError):
            win_dur = 1.0
        try:
            overlap = float(overlap)
        except (TypeError, ValueError):
            overlap = 0.5
        overlap = max(0.0, min(1.0, overlap))
        hop_seconds = max(0.0, win_dur * (1.0 - overlap))
        return max(win_dur * 0.5, hop_seconds * 0.5)

    try:
        return float(clip_pad_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("clip_pad_seconds must be a non-negative float or 'auto'") from exc


def create_custom_spectrograms_from_json(
    self,
    json_path: str,
    *,
    generator_options: Optional[Dict[str, Any]] = None,
    clip_pad_seconds: Union[float, str, None] = 'auto',
    default_pad_seconds: float = 0.0,
    default_tag: str = 'custom_spectrograms',
    audio_download_workers: Optional[int] = None,
    custom_output_dir: Optional[str] = None,
    save_png: bool = False,
    save_mat: bool = True,
    save_npy: bool = False,
    save_plot: Optional[bool] = None,
    save_context_audio: bool = False,
) -> List[Dict[str, Any]]:
    """Download audio clips from a JSON request file and generate custom spectrograms locally.

    ``pad_seconds`` in the JSON defines the target clip around each timestamp.
    ``clip_pad_seconds`` adds extra context (seconds) to the downloaded audio to
    reduce edge artifacts during STFT computation. When ``freq_lims`` are provided
    in generator options, the saved outputs are cropped to that range.
    The final saved audio clip is trimmed to the requested window length
    (the extra context is used only for spectrogram generation). Use
    ``save_context_audio=True`` to keep the intermediate context clip.

    Args:
        json_path: Path to the JSON request file.
        generator_options: Default spectrogram generator settings applied to all requests.
        clip_pad_seconds: Extra context seconds around the clip. Use ``'auto'`` to
            derive from STFT settings.
        default_pad_seconds: Default padding if not provided per request.
        default_tag: Output folder tag if not provided per request.
        audio_download_workers: Override parallel audio download workers.
        custom_output_dir: Optional custom output directory for generated spectrograms.
        save_png: Save PNG plot outputs (default: False).
        save_mat: Save MAT outputs (default: True).
        save_npy: Save NumPy outputs (default: False).
        save_plot: Legacy alias for ``save_png``.
        save_context_audio: If True, keep the intermediate context clip used
            for spectrogram generation.

    Returns:
        List of per‑request summary dicts with audio and custom spectrogram outputs.

    JSON supports generator defaults and per‑request overrides::

        {
          \"defaults\": {...},
          \"generator_defaults\": {\"win_dur\": 0.5, \"overlap\": 0.5},
          \"requests\": [
            {
              \"timestamp\": \"...\",
              \"generator_options\": {\"freq_lims\": [10, 1000]}
            }
          ]
        }
    """
    from ...audio import SpectrogramGenerator

    def normalize_generator_options(raw: Any, label: str) -> Dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            self.logger.warning(f"{label} generator options must be a dict; skipping")
            return {}
        cleaned = dict(raw)
        for key in ('clip_start', 'clip_end', 'clip_pad_seconds'):
            if key in cleaned:
                cleaned.pop(key, None)
                self.logger.warning(f"Ignoring {key} in {label} generator options; it is set per-request.")
        return cleaned

    if save_plot is not None:
        save_png = bool(save_plot)

    with open(json_path, 'r') as f:
        payload = json.load(f)

    base_generator_options = normalize_generator_options(generator_options, "function")
    json_generator_defaults: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ('generator_defaults', 'spectrogram_generator_defaults', 'custom_spectrogram_defaults'):
            if key in payload:
                json_generator_defaults = normalize_generator_options(payload.get(key), f"JSON {key}")
                break

    request_items: List[Tuple[TimestampRequest, Dict[str, Any]]] = []
    if isinstance(payload, dict) and 'requests' in payload:
        defaults = payload.get('defaults', {})
        for entry in payload.get('requests', []):
            if not isinstance(entry, dict):
                continue
            entry_generator = {}
            for key in ('generator_options', 'spectrogram_generator_options', 'custom_spectrogram_options'):
                if key in entry:
                    entry_generator = normalize_generator_options(entry.get(key), f"request {key}")
                    break
            merged_generator = {
                **base_generator_options,
                **json_generator_defaults,
                **entry_generator,
            }
            if 'crop_freq_lims' not in merged_generator:
                merged_generator['crop_freq_lims'] = 'freq_lims' in merged_generator
            req = self._build_request_from_dict(
                entry,
                defaults=defaults,
                default_pad_seconds=default_pad_seconds,
                default_tag=default_tag,
                clip_outputs=True,
                spectrogram_format='mat',
                download_audio=True,
                download_spectrogram=False,
            )
            request_items.append((req, merged_generator))
    else:
        requests = self._coerce_timestamp_requests(
            payload,
            default_pad_seconds=default_pad_seconds,
            default_tag=default_tag,
            clip_outputs=True,
            spectrogram_format='mat',
            download_audio=True,
            download_spectrogram=False,
        )
        merged_defaults = {**base_generator_options, **json_generator_defaults}
        if 'crop_freq_lims' not in merged_defaults:
            merged_defaults['crop_freq_lims'] = 'freq_lims' in merged_defaults
        request_items = [(req, merged_defaults) for req in requests]

    summaries: List[Dict[str, Any]] = []
    for request, gen_options in request_items:
        context_seconds = self._resolve_custom_clip_pad_seconds(clip_pad_seconds, gen_options)
        if context_seconds < 0:
            raise ValueError("clip_pad_seconds must be >= 0")

        base_pad_before = float(request.pad_before or 0.0)
        base_pad_after = float(request.pad_after or 0.0)
        target_duration = base_pad_before + base_pad_after
        if target_duration <= 0:
            target_duration = 1.0

        extended_request = replace(
            request,
            pad_before=base_pad_before + context_seconds,
            pad_after=base_pad_after + context_seconds,
            want_audio=True,
            want_spectrogram=False,
            clip_outputs=True,
        )
        summary = self._execute_timestamp_request(
            extended_request,
            audio_download_workers=audio_download_workers,
        )

        audio_clip = (summary.get('audio') or {}).get('clip_path')
        custom_spec: Dict[str, Any] = {}
        context_audio_clip = audio_clip
        final_audio_clip: Optional[str] = None
        keep_context_audio = False
        if context_audio_clip:
            clip_start_dt = ensure_timezone_aware(request.timestamp) - timedelta(seconds=base_pad_before)
            clip_end_dt = ensure_timezone_aware(request.timestamp) + timedelta(seconds=base_pad_after)
            trimmed_audio = self._trim_audio_clip(
                context_audio_clip,
                request,
                clip_start=clip_start_dt,
                clip_end=clip_end_dt,
                offset_seconds=context_seconds,
                duration_seconds=target_duration,
            )
            if trimmed_audio:
                final_audio_clip = trimmed_audio
                keep_context_audio = bool(save_context_audio)
                summary_audio = summary.get('audio') or {}
                if keep_context_audio:
                    summary_audio['context_clip_path'] = context_audio_clip
                summary_audio['clip_path'] = trimmed_audio
                summary['audio'] = summary_audio
            else:
                final_audio_clip = context_audio_clip
                keep_context_audio = True
                self.logger.warning(
                    f"Using context clip for audio output; failed to trim {context_audio_clip}"
                )
        if context_audio_clip:
            if custom_output_dir:
                output_dir = Path(custom_output_dir)
            else:
                output_dir = Path(self.audio_path).parent / "custom_spectrograms"
            output_dir.mkdir(parents=True, exist_ok=True)

            generator = SpectrogramGenerator(
                **gen_options,
                clip_start=context_seconds,
                clip_end=context_seconds + target_duration,
                clip_pad_seconds=context_seconds,
            )
            extra_metadata = {
                'requested_timestamp': request.timestamp.isoformat(),
                'requested_window_seconds': {
                    'pad_before': base_pad_before,
                    'pad_after': base_pad_after,
                    'duration': target_duration,
                },
                'context_seconds': context_seconds,
                'downloaded_audio_files': (summary.get('audio') or {}).get('files', []),
                'final_audio_clip': final_audio_clip,
            }
            if keep_context_audio:
                extra_metadata['context_audio_clip'] = context_audio_clip
            try:
                result = generator.process_single_file(
                    context_audio_clip,
                    output_dir,
                    save_plot=save_png,
                    save_mat=save_mat,
                    save_npy=save_npy,
                    extra_metadata=extra_metadata,
                )
                custom_spec = {
                    'audio_file': result.get('audio_file'),
                    'mat_file': result.get('mat_file'),
                    'png_file': result.get('png_file'),
                    'npy_file': result.get('npy_file'),
                    'duration': result.get('duration'),
                    'sample_rate': result.get('sample_rate'),
                    'clip_start_seconds': context_seconds,
                    'clip_end_seconds': context_seconds + target_duration,
                    'clip_pad_seconds': context_seconds,
                    'freq_lims': gen_options.get('freq_lims'),
                    'log_freq': gen_options.get('log_freq', True),
                    'crop_freq_lims': gen_options.get('crop_freq_lims', True),
                    'output_dir': str(output_dir),
                }
                if final_audio_clip:
                    custom_spec['final_audio_clip'] = final_audio_clip
                if keep_context_audio and context_audio_clip:
                    custom_spec['context_audio_clip'] = context_audio_clip
            except Exception as exc:
                custom_spec = {
                    'error': str(exc),
                    'output_dir': str(output_dir),
                }
            if context_audio_clip and not keep_context_audio:
                try:
                    os.remove(context_audio_clip)
                except OSError:
                    pass
                context_json = os.path.splitext(context_audio_clip)[0] + ".json"
                try:
                    os.remove(context_json)
                except OSError:
                    pass
        else:
            custom_spec = {
                'error': 'No audio clip produced for request.',
            }

        summary['custom_spectrogram'] = custom_spec
        summaries.append(summary)

    return summaries
