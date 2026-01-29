#!/usr/bin/env python3
"""
Benchmark ONC audio download methods for a given device/time range.

Methods:
  - list-seq: getListByDevice + sequential getFile
  - list-par: getListByDevice + parallel getFile
  - data-product: requestDataProduct/runDataProduct/downloadDataProduct (HAF)
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from onc.onc import ONC
from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator
from onc_hydrophone_data.onc.common import load_config, ensure_timezone_aware, format_iso_utc


def parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    return ensure_timezone_aware(dt)


def list_audio_files(onc: ONC, device: str, start_iso: str, end_iso: str, extension: str):
    filters = {
        "deviceCode": device,
        "dateFrom": start_iso,
        "dateTo": end_iso,
        "extension": extension,
    }
    result = onc.getListByDevice(filters, allPages=True)
    files = result.get("files", []) if isinstance(result, dict) else []
    return [f for f in files if f.lower().endswith(f".{extension}")]


def download_files(onc: ONC, files, out_dir: str, *, parallel: bool, max_workers: int):
    os.makedirs(out_dir, exist_ok=True)
    original_out = onc.outPath
    onc.outPath = out_dir
    errors = 0
    start = time.perf_counter()
    try:
        if parallel and len(files) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(onc.getFile, f, overwrite=True): f for f in files}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        errors += 1
        else:
            for f in files:
                try:
                    onc.getFile(f, overwrite=True)
                except Exception:
                    errors += 1
    finally:
        onc.outPath = original_out
    elapsed = time.perf_counter() - start
    bytes_downloaded = 0
    downloaded = 0
    for f in files:
        local_path = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(local_path):
            downloaded += 1
            bytes_downloaded += os.path.getsize(local_path)
    return elapsed, bytes_downloaded, downloaded, errors


def run_list_method(onc, device, start_iso, end_iso, extension, max_files, out_dir, parallel, max_workers):
    files = list_audio_files(onc, device, start_iso, end_iso, extension)
    selected = files[:max_files] if max_files and max_files > 0 else files
    if not selected:
        return {
            "method": "list-par" if parallel else "list-seq",
            "extension": extension,
            "files_found": len(files),
            "files_requested": 0,
            "files_downloaded": 0,
            "bytes_downloaded": 0,
            "seconds": 0.0,
            "errors": 0,
        }
    elapsed, bytes_downloaded, downloaded, errors = download_files(
        onc,
        selected,
        out_dir,
        parallel=parallel,
        max_workers=max_workers,
    )
    return {
        "method": "list-par" if parallel else "list-seq",
        "extension": extension,
        "files_found": len(files),
        "files_requested": len(selected),
        "files_downloaded": downloaded,
        "bytes_downloaded": bytes_downloaded,
        "seconds": elapsed,
        "errors": errors,
    }


def run_data_product_method(onc, device, start_iso, end_iso, extension, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filters = {
        "dataProductCode": "HAF",
        "deviceCode": device,
        "dateFrom": start_iso,
        "dateTo": end_iso,
    }
    if extension:
        filters["extension"] = extension
    start = time.perf_counter()
    try:
        result = onc.requestDataProduct(filters)
        dp_request_id = result["dpRequestId"] if isinstance(result, dict) else result
        run_data = onc.runDataProduct(dp_request_id, waitComplete=True)
        run_ids = run_data.get("runIds") if isinstance(run_data, dict) else None
        if not run_ids:
            raise RuntimeError("No runIds returned for data product request")
        original_out = onc.outPath
        onc.outPath = out_dir
        try:
            onc.downloadDataProduct(run_ids[0])
        finally:
            onc.outPath = original_out
    except Exception as exc:
        return {
            "method": "data-product",
            "extension": extension,
            "files_found": 0,
            "files_requested": 0,
            "files_downloaded": 0,
            "bytes_downloaded": 0,
            "seconds": time.perf_counter() - start,
            "errors": 1,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - start
    bytes_downloaded = 0
    downloaded = 0
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            downloaded += 1
            bytes_downloaded += os.path.getsize(path)
    return {
        "method": "data-product",
        "extension": extension,
        "files_found": downloaded,
        "files_requested": downloaded,
        "files_downloaded": downloaded,
        "bytes_downloaded": bytes_downloaded,
        "seconds": elapsed,
        "errors": 0,
    }


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def _collect_audio_files(input_path: str, max_files: int):
    path = Path(input_path).expanduser()
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        files.sort()
    else:
        files = [path]
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def _parse_spectrogram_backend(label: str):
    normalized = label.strip().lower()
    if normalized in {"scipy", "scipy-cpu"}:
        return {"label": label, "backend": "scipy", "torch_device": None}
    if normalized in {"torch", "torch-cpu"}:
        return {"label": label, "backend": "torch", "torch_device": "cpu"}
    if normalized in {"torch-gpu", "torch-cuda", "cuda", "gpu"}:
        return {"label": label, "backend": "torch", "torch_device": "cuda"}
    raise ValueError(f"Unknown spectrogram backend '{label}' (expected scipy, torch-cpu, torch-gpu)")


def _summarize_timings(timings):
    if not timings:
        return None
    return {
        "count": len(timings),
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }


def run_spectrogram_benchmark(
    input_path: str,
    backends,
    repeats: int,
    warmup: int,
    max_files: int,
    max_duration: Optional[float],
):
    files = _collect_audio_files(input_path, max_files)
    if not files:
        return {"error": f"No audio files found under {input_path}"}

    try:
        import torch  # Optional dependency for torch backends
    except Exception:
        torch = None

    loader = SpectrogramGenerator(
        max_duration=max_duration,
        backend="scipy",
        quiet=True,
        use_logging=False,
    )

    audio_batches = []
    for audio_path in files:
        audio_data, sample_rate, clip_meta = loader.load_audio(audio_path)
        audio_batches.append(
            {
                "path": str(audio_path),
                "samples": int(len(audio_data)),
                "sample_rate": int(sample_rate),
                "duration_seconds": float(len(audio_data) / sample_rate),
                "audio_data": audio_data,
                "clip_meta": clip_meta,
            }
        )

    backend_specs = [_parse_spectrogram_backend(label) for label in backends]
    results = []

    for spec in backend_specs:
        backend = spec["backend"]
        torch_device = spec["torch_device"]
        label = spec["label"]

        if backend == "torch" and torch is None:
            results.append(
                {
                    "label": label,
                    "backend": backend,
                    "torch_device": torch_device,
                    "error": "torch not installed in this environment",
                }
            )
            continue

        if backend == "torch" and torch_device and torch_device.startswith("cuda"):
            if not torch.cuda.is_available():
                results.append(
                    {
                        "label": label,
                        "backend": backend,
                        "torch_device": torch_device,
                        "error": "CUDA not available in this environment",
                    }
                )
                continue

        generator = SpectrogramGenerator(
            max_duration=max_duration,
            backend=backend,
            torch_device=torch_device or "cpu",
            quiet=True,
            use_logging=False,
        )

        timings = []
        for batch in audio_batches:
            audio_data = batch["audio_data"]
            sample_rate = batch["sample_rate"]
            clip_meta = batch["clip_meta"]
            for idx in range(warmup + repeats):
                if backend == "torch" and torch_device and torch_device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                generator.compute_spectrogram(audio_data, sample_rate, clip_meta)
                if backend == "torch" and torch_device and torch_device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if idx >= warmup:
                    timings.append(elapsed)

        summary = _summarize_timings(timings)
        result = {
            "label": label,
            "backend": backend,
            "torch_device": torch_device,
            "timing": summary,
        }
        if torch is not None and backend == "torch":
            result["torch_version"] = torch.__version__
            result["cuda_available"] = torch.cuda.is_available()
        results.append(result)

    return {
        "input_path": str(input_path),
        "files": [
            {
                "path": b["path"],
                "samples": b["samples"],
                "sample_rate": b["sample_rate"],
                "duration_seconds": b["duration_seconds"],
            }
            for b in audio_batches
        ],
        "repeats": repeats,
        "warmup": warmup,
        "max_duration": max_duration,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONC audio download methods and spectrogram generation.")
    parser.add_argument("--device", default="ICLISTENHF6324", help="ONC device code")
    parser.add_argument("--start", default="2024-04-01T12:00:00Z", help="Start ISO time")
    parser.add_argument("--end", default="2024-04-01T12:10:00Z", help="End ISO time")
    parser.add_argument(
        "--formats",
        default="flac,wav",
        help="Comma-separated extensions to test (default: flac,wav)",
    )
    parser.add_argument(
        "--methods",
        default="list-seq,list-par",
        help="Comma-separated methods: list-seq,list-par,data-product",
    )
    parser.add_argument("--max-files", type=int, default=2, help="Limit number of files per method")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel worker count")
    parser.add_argument("--output-dir", default=None, help="Base output directory")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip download benchmarks (useful for spectrogram-only runs).",
    )
    parser.add_argument(
        "--bench-spectrograms",
        action="store_true",
        help="Run spectrogram generation benchmarks.",
    )
    parser.add_argument(
        "--spectrogram-input",
        default=None,
        help="Audio file or directory for spectrogram benchmarks (default: test.wav if present).",
    )
    parser.add_argument(
        "--spectrogram-backends",
        default="torch-cpu,torch-gpu,scipy",
        help="Comma-separated spectrogram backends: torch-cpu, torch-gpu, scipy",
    )
    parser.add_argument(
        "--spectrogram-repeats",
        type=int,
        default=5,
        help="Number of timed spectrogram runs per backend.",
    )
    parser.add_argument(
        "--spectrogram-warmup",
        type=int,
        default=1,
        help="Warmup runs per backend (not included in timings).",
    )
    parser.add_argument(
        "--spectrogram-max-files",
        type=int,
        default=1,
        help="Max number of audio files to benchmark.",
    )
    parser.add_argument(
        "--spectrogram-max-duration",
        type=float,
        default=None,
        help="Max duration (seconds) to load from each audio file.",
    )
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {}

    run_downloads = (not args.skip_downloads) and bool(methods)
    if run_downloads:
        onc_token, data_dir = load_config()
        onc = ONC(onc_token, showInfo=False)

        start_dt = parse_iso_datetime(args.start)
        end_dt = parse_iso_datetime(args.end)
        start_iso = format_iso_utc(start_dt)
        end_iso = format_iso_utc(end_dt)

        base_output = args.output_dir or os.path.join(
            data_dir,
            "bench_audio",
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        )
        os.makedirs(base_output, exist_ok=True)

        download_results = {
            "device": args.device,
            "start": start_iso,
            "end": end_iso,
            "output_dir": base_output,
            "results": [],
        }

        for extension in formats:
            if "list-seq" in methods:
                out_dir = os.path.join(base_output, f"list-seq_{extension}")
                download_results["results"].append(
                    run_list_method(
                        onc,
                        args.device,
                        start_iso,
                        end_iso,
                        extension,
                        args.max_files,
                        out_dir,
                        parallel=False,
                        max_workers=args.max_workers,
                    )
                )
            if "list-par" in methods:
                out_dir = os.path.join(base_output, f"list-par_{extension}")
                download_results["results"].append(
                    run_list_method(
                        onc,
                        args.device,
                        start_iso,
                        end_iso,
                        extension,
                        args.max_files,
                        out_dir,
                        parallel=True,
                        max_workers=args.max_workers,
                    )
                )
            if "data-product" in methods:
                out_dir = os.path.join(base_output, f"data-product_{extension}")
                download_results["results"].append(
                    run_data_product_method(
                        onc,
                        args.device,
                        start_iso,
                        end_iso,
                        extension,
                        out_dir,
                    )
                )

        results["download_benchmark"] = download_results

    if args.bench_spectrograms:
        spectrogram_input = args.spectrogram_input
        if spectrogram_input is None:
            default_audio = os.path.join(REPO_ROOT, "test.wav")
            if os.path.exists(default_audio):
                spectrogram_input = default_audio
            else:
                spectrogram_input = REPO_ROOT

        backend_labels = [b.strip() for b in args.spectrogram_backends.split(",") if b.strip()]
        results["spectrogram_benchmark"] = run_spectrogram_benchmark(
            spectrogram_input,
            backend_labels,
            repeats=args.spectrogram_repeats,
            warmup=args.spectrogram_warmup,
            max_files=args.spectrogram_max_files,
            max_duration=args.spectrogram_max_duration,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
