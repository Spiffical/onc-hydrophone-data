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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from onc.onc import ONC
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONC audio download methods.")
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
    args = parser.parse_args()

    onc_token, data_dir = load_config()
    onc = ONC(onc_token, showInfo=False)

    start_dt = parse_iso_datetime(args.start)
    end_dt = parse_iso_datetime(args.end)
    start_iso = format_iso_utc(start_dt)
    end_iso = format_iso_utc(end_dt)

    base_output = args.output_dir or os.path.join(data_dir, "bench_audio", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    os.makedirs(base_output, exist_ok=True)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    results = {
        "device": args.device,
        "start": start_iso,
        "end": end_iso,
        "output_dir": base_output,
        "results": [],
    }

    for extension in formats:
        if "list-seq" in methods:
            out_dir = os.path.join(base_output, f"list-seq_{extension}")
            results["results"].append(
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
            results["results"].append(
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
            results["results"].append(
                run_data_product_method(
                    onc,
                    args.device,
                    start_iso,
                    end_iso,
                    extension,
                    out_dir,
                )
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
