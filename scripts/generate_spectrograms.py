#!/usr/bin/env python3
"""
Generate custom spectrograms from audio files (FLAC, WAV, etc.).

This script provides functionality to create spectrograms from audio files using configurable
parameters. It's designed to work with the existing project structure and supports both
interactive and batch processing modes.

Key Features:
- Support for multiple audio formats (FLAC, WAV, MP3, M4A)
- Configurable spectrogram parameters (window duration, overlap, frequency limits)
- Batch processing of directories
- MATLAB-compatible output format
- Integration with existing data folder structure

Usage Examples:

  # Interactive mode (recommended for beginners)
  python scripts/generate_spectrograms.py

  # Process a specific directory
  python scripts/generate_spectrograms.py --input-dir data/ICLISTENHF6020/flac/ --output-dir data/ICLISTENHF6020/custom_spectrograms/

  # Custom parameters
  python scripts/generate_spectrograms.py --input-dir audio/ --win-dur 2.0 --overlap 0.75 --freq-min 5 --freq-max 20000

  # Process single file
  python scripts/generate_spectrograms.py --input-file audio.flac --output-dir spectrograms/
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onc_hydrophone_data.audio import SpectrogramGenerator, find_audio_files, get_audio_info, estimate_processing_time
from onc_hydrophone_data.data.config_utils import DatasetConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(message: str, status: str = "INFO"):
    """Print a status message with formatting."""
    colors = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green  
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "PROGRESS": "\033[96m"   # Cyan
    }
    reset = "\033[0m"
    color = colors.get(status, colors["INFO"])
    prefix = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄"
    }.get(status, "ℹ️")
    
    print(f"{color}{prefix} {message}{reset}")

def prompt_for_input_source() -> tuple:
    """
    Prompt user to select input source (directory or file).
    Returns tuple of (input_path, is_directory)
    """
    print_header("INPUT SOURCE SELECTION")
    print("Choose your input source:")
    print("1. Process a directory of audio files")
    print("2. Process a single audio file")
    print("3. Auto-detect from project structure")
    print("4. Cancel")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            return None, None
        elif choice == '1':
            # Directory input
            while True:
                dir_path = input("Enter directory path containing audio files: ").strip()
                if not dir_path:
                    print_status("Input cancelled", "WARNING")
                    return None, None
                
                dir_path = Path(dir_path)
                if dir_path.exists():
                    audio_files = find_audio_files(dir_path)
                    if audio_files:
                        print_status(f"Found {len(audio_files)} audio files in {dir_path}", "SUCCESS")
                        return str(dir_path), True
                    else:
                        print_status("No audio files found in directory", "WARNING")
                        continue
                else:
                    print_status("Directory not found", "ERROR")
                    continue
                    
        elif choice == '2':
            # Single file input
            while True:
                file_path = input("Enter audio file path: ").strip()
                if not file_path:
                    print_status("Input cancelled", "WARNING")
                    return None, None
                
                file_path = Path(file_path)
                if file_path.exists() and file_path.is_file():
                    print_status(f"Selected file: {file_path}", "SUCCESS")
                    return str(file_path), False
                else:
                    print_status("File not found", "ERROR")
                    continue
                    
        elif choice == '3':
            # Auto-detect from project structure
            project_dirs = []
            data_dir = Path("data")
            if data_dir.exists():
                for device_dir in data_dir.iterdir():
                    if device_dir.is_dir():
                        # Look for audio directories (new structure) and flac (legacy)
                        audio_dirs = list(device_dir.rglob("audio")) + list(device_dir.rglob("flac"))
                        for audio_dir in audio_dirs:
                            audio_files = find_audio_files(audio_dir)
                            if audio_files:
                                project_dirs.append((audio_dir, len(audio_files)))
            
            if not project_dirs:
                print_status("No audio directories found in project structure", "WARNING")
                continue
            
            print("\nFound audio directories:")
            for i, (dir_path, file_count) in enumerate(project_dirs, 1):
                print(f"{i}. {dir_path} ({file_count} files)")
            
            try:
                dir_choice = int(input(f"\nSelect directory (1-{len(project_dirs)}): ").strip())
                if 1 <= dir_choice <= len(project_dirs):
                    selected_dir = project_dirs[dir_choice - 1][0]
                    print_status(f"Selected: {selected_dir}", "SUCCESS")
                    return str(selected_dir), True
                else:
                    print_status("Invalid selection", "ERROR")
                    continue
            except ValueError:
                print_status("Please enter a valid number", "ERROR")
                continue
        else:
            print_status("Please enter a number between 1 and 4", "WARNING")
            continue

def parse_window_type(value: Optional[str]):
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {'none', 'null'}:
        return None
    for sep in (',', ':'):
        if sep in cleaned:
            name, param = cleaned.split(sep, 1)
            name = name.strip()
            param = param.strip()
            if not name:
                return None
            if param:
                try:
                    return (name, float(param))
                except ValueError:
                    return cleaned
    return cleaned

def _prompt_optional_float(prompt: str) -> Optional[float]:
    while True:
        value = input(prompt).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            print_status("Please enter a valid number", "ERROR")

def _prompt_optional_int(prompt: str) -> Optional[int]:
    while True:
        value = input(prompt).strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            print_status("Please enter a valid integer", "ERROR")

def _prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        value = input(f\"{prompt}{suffix}: \").strip().lower()
        if not value:
            return default
        if value in {'y', 'yes'}:
            return True
        if value in {'n', 'no'}:
            return False
        print_status("Please enter y or n", "ERROR")

def prompt_for_parameters() -> dict:
    """
    Prompt user for spectrogram parameters.
    Returns dictionary of parameters.
    """
    print_header("SPECTROGRAM PARAMETERS")
    print("Configure spectrogram generation parameters:")
    print("(Press Enter to use default values)")
    
    params = {}
    
    # Window duration
    while True:
        win_dur_input = input("Window duration in seconds [default: 1.0]: ").strip()
        if not win_dur_input:
            params['win_dur'] = 1.0
            break
        try:
            params['win_dur'] = float(win_dur_input)
            if params['win_dur'] > 0:
                break
            else:
                print_status("Window duration must be positive", "ERROR")
        except ValueError:
            print_status("Please enter a valid number", "ERROR")
    
    # Overlap
    while True:
        overlap_input = input("Overlap ratio (0-1) [default: 0.5]: ").strip()
        if not overlap_input:
            params['overlap'] = 0.5
            break
        try:
            params['overlap'] = float(overlap_input)
            if 0 <= params['overlap'] < 1:
                break
            else:
                print_status("Overlap must be between 0 and 1", "ERROR")
        except ValueError:
            print_status("Please enter a valid number", "ERROR")

    window_input = input(
        "Window type [default: hann] (e.g., hann, hamming, kaiser,14): "
    ).strip()
    params['window_type'] = parse_window_type(window_input) or 'hann'

    params['nfft'] = _prompt_optional_int("FFT size nfft in samples [default: auto]: ")
    params['win_length'] = _prompt_optional_int("Window length in samples [default: auto]: ")
    params['hop_length'] = _prompt_optional_int(
        "Hop length in samples [default: auto] (overrides overlap): "
    )
    
    # Frequency limits
    while True:
        freq_min_input = input("Minimum frequency (Hz) [default: 10]: ").strip()
        if not freq_min_input:
            params['freq_min'] = 10
            break
        try:
            params['freq_min'] = float(freq_min_input)
            if params['freq_min'] > 0:
                break
            else:
                print_status("Frequency must be positive", "ERROR")
        except ValueError:
            print_status("Please enter a valid number", "ERROR")
    
    while True:
        freq_max_input = input("Maximum frequency (Hz) [default: 10000]: ").strip()
        if not freq_max_input:
            params['freq_max'] = 10000
            break
        try:
            params['freq_max'] = float(freq_max_input)
            if params['freq_max'] > params['freq_min']:
                break
            else:
                print_status(f"Maximum frequency must be greater than {params['freq_min']}", "ERROR")
        except ValueError:
            print_status("Please enter a valid number", "ERROR")

    colormap_input = input("Colormap [default: turbo]: ").strip()
    params['colormap'] = colormap_input or 'turbo'

    while True:
        clim_min_input = input("Color scale minimum dB [default: -60]: ").strip()
        if not clim_min_input:
            params['clim_min'] = -60
            break
        try:
            params['clim_min'] = float(clim_min_input)
            break
        except ValueError:
            print_status("Please enter a valid number", "ERROR")

    while True:
        clim_max_input = input("Color scale maximum dB [default: 0]: ").strip()
        if not clim_max_input:
            params['clim_max'] = 0
            break
        try:
            params['clim_max'] = float(clim_max_input)
            if params['clim_max'] > params['clim_min']:
                break
            print_status("Maximum dB must be greater than minimum dB", "ERROR")
        except ValueError:
            print_status("Please enter a valid number", "ERROR")

    params['log_freq'] = _prompt_yes_no("Log frequency axis?", default=True)
    params['max_duration'] = _prompt_optional_float(
        "Maximum duration per file in seconds [default: full file]: "
    )
    params['clip_start'] = _prompt_optional_float("Clip start time in seconds [default: none]: ")
    params['clip_end'] = _prompt_optional_float("Clip end time in seconds [default: none]: ")
    params['clip_pad_seconds'] = _prompt_optional_float(
        "Clip context padding in seconds on each side [default: auto]: "
    )

    while True:
        backend_input = input("Backend [default: auto] (auto/torch/scipy): ").strip().lower()
        if not backend_input:
            params['backend'] = 'auto'
            break
        if backend_input in {'auto', 'torch', 'scipy'}:
            params['backend'] = backend_input
            break
        print_status("Please enter auto, torch, or scipy", "ERROR")

    while True:
        scaling_input = input("Scaling [default: density] (density/spectrum): ").strip().lower()
        if not scaling_input:
            params['scaling'] = 'density'
            break
        if scaling_input in {'density', 'spectrum'}:
            params['scaling'] = scaling_input
            break
        print_status("Please enter density or spectrum", "ERROR")

    params['quiet'] = _prompt_yes_no("Quiet mode (reduced logging)?", default=False)
    params['use_logging'] = _prompt_yes_no("Use logging module?", default=True)
    
    # Output format
    print("\nOutput format options:")
    print("1. MATLAB files only (.mat)")
    print("2. PNG plots only (.png)")
    print("3. Both MATLAB and PNG files")
    
    while True:
        format_choice = input("Choose output format (1-3) [default: 3]: ").strip()
        if not format_choice:
            params['save_mat'] = True
            params['save_plot'] = True
            break
        elif format_choice == '1':
            params['save_mat'] = True
            params['save_plot'] = False
            break
        elif format_choice == '2':
            params['save_mat'] = False
            params['save_plot'] = True
            break
        elif format_choice == '3':
            params['save_mat'] = True
            params['save_plot'] = True
            break
        else:
            print_status("Please enter 1, 2, or 3", "ERROR")
    
    print_status("Parameters configured successfully", "SUCCESS")
    return params

def determine_output_directory(input_path: str, is_directory: bool, output_dir: Optional[str] = None) -> Path:
    """
    Determine appropriate output directory based on input and project structure.
    """
    if output_dir:
        return Path(output_dir)
    
    input_path = Path(input_path)
    
    if is_directory:
        # If input is in project structure (data/DEVICE/METHOD/flac/), 
        # create parallel custom_spectrograms directory
        if input_path.parts[:2] == ('data',) and len(input_path.parts) >= 4:
            # Structure: data/DEVICE/METHOD/flac -> data/DEVICE/METHOD/custom_spectrograms
            output_path = input_path.parent / "custom_spectrograms"
        else:
            # Generic output directory
            output_path = input_path.parent / "spectrograms"
    else:
        # Single file - create output directory next to file
        output_path = input_path.parent / "spectrograms"
    
    return output_path

def process_audio_files(input_path: str, output_dir: str, is_directory: bool, 
                       generator: SpectrogramGenerator, save_mat: bool, save_plot: bool) -> dict:
    """
    Process audio files and generate spectrograms.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    if is_directory:
        audio_files = find_audio_files(input_path)
        if not audio_files:
            raise ValueError(f"No audio files found in {input_path}")
        
        print_status(f"Processing {len(audio_files)} files from {input_path}", "PROGRESS")
        
        # Estimate processing time
        estimated_time = estimate_processing_time(audio_files, generator.win_dur)
        if estimated_time > 60:
            print_status(f"Estimated processing time: {estimated_time/60:.1f} minutes", "INFO")
        else:
            print_status(f"Estimated processing time: {estimated_time:.1f} seconds", "INFO")
        
        # Process all files
        results = generator.process_directory(input_path, output_dir, save_mat=save_mat, save_plot=save_plot)
        
    else:
        # Single file
        print_status(f"Processing single file: {input_path.name}", "PROGRESS")
        result = generator.process_single_file(input_path, output_dir, save_mat=save_mat, save_plot=save_plot)
        results = [result]
    
    # Summarize results
    successful = len([r for r in results if 'error' not in r])
    failed = len(results) - successful
    
    summary = {
        'total_files': len(results),
        'successful': successful,
        'failed': failed,
        'output_directory': str(output_dir),
        'results': results
    }
    
    return summary

def main():
    """Main function to handle command line arguments and run spectrogram generation."""
    
    parser = argparse.ArgumentParser(
        description='Generate custom spectrograms from audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python %(prog)s
  
  # Process directory with default settings
  python %(prog)s --input-dir data/ICLISTENHF6020/flac/
  
  # Custom parameters
  python %(prog)s --input-dir audio/ --win-dur 2.0 --overlap 0.75 --freq-min 5 --freq-max 20000
  
  # Single file processing
  python %(prog)s --input-file audio.flac --output-dir spectrograms/
  
  # MATLAB files only
  python %(prog)s --input-dir audio/ --no-plots
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-dir', 
                            help='Directory containing audio files')
    input_group.add_argument('--input-file', 
                            help='Single audio file to process')
    
    # Output options
    parser.add_argument('--output-dir',
                       help='Output directory for spectrograms (auto-determined if not specified)')
    
    # Spectrogram parameters
    parser.add_argument('--win-dur', 
                       type=float, 
                       default=1.0,
                       help='Window duration in seconds (default: 1.0)')
    
    parser.add_argument('--overlap', 
                       type=float, 
                       default=0.5,
                       help='Overlap ratio between windows 0-1 (default: 0.5)')

    parser.add_argument('--window-type',
                       default='hann',
                       help='Window type (e.g., hann, hamming, kaiser,14) (default: hann)')

    parser.add_argument('--nfft',
                       type=int,
                       default=None,
                       help='FFT size in samples (default: derived from win_dur)')

    parser.add_argument('--win-length',
                       type=int,
                       default=None,
                       help='Window length in samples (default: nfft)')

    parser.add_argument('--hop-length',
                       type=int,
                       default=None,
                       help='Hop length in samples (overrides overlap)')
    
    parser.add_argument('--max-duration', 
                       type=float, 
                       default=None,
                       help='Maximum duration to process in seconds (default: process full file)')

    parser.add_argument('--clip-start',
                       type=float,
                       default=None,
                       help='Clip start time in seconds (default: none)')

    parser.add_argument('--clip-end',
                       type=float,
                       default=None,
                       help='Clip end time in seconds (default: none)')

    parser.add_argument('--clip-pad-seconds',
                       type=float,
                       default=None,
                       help='Extra context (seconds) on each side before STFT (default: auto)')
    
    parser.add_argument('--freq-min', 
                       type=float, 
                       default=10,
                       help='Minimum frequency for plots in Hz (default: 10)')
    
    parser.add_argument('--freq-max', 
                       type=float, 
                       default=10000,
                       help='Maximum frequency for plots in Hz (default: 10000)')
    
    # Output format options
    parser.add_argument('--no-plots', 
                       action='store_true',
                       help='Skip PNG plot generation (MATLAB files only)')
    
    parser.add_argument('--no-mat', 
                       action='store_true',
                       help='Skip MATLAB file generation (PNG plots only)')
    
    parser.add_argument('--colormap', 
                       default='turbo',
                       help='Matplotlib colormap for plots (default: turbo)')
    
    parser.add_argument('--log-freq', 
                       action='store_true',
                       default=True,
                       help='Use logarithmic frequency scale (default: True)')

    parser.add_argument('--linear-freq',
                       dest='log_freq',
                       action='store_false',
                       help='Use linear frequency scale')
    
    parser.add_argument('--clim-min', 
                       type=float, 
                       default=-60,
                       help='Minimum color scale value in dB (default: -60)')
    
    parser.add_argument('--clim-max', 
                       type=float, 
                       default=0,
                       help='Maximum color scale value in dB (default: 0)')

    parser.add_argument('--backend',
                       choices=['auto', 'torch', 'scipy'],
                       default='auto',
                       help='Spectrogram backend: auto, torch, or scipy (default: auto)')

    parser.add_argument('--scaling',
                       choices=['density', 'spectrum'],
                       default='density',
                       help='PSD scaling: density or spectrum (default: density)')

    parser.add_argument('--quiet',
                       action='store_true',
                       default=False,
                       help='Reduce logging output')

    parser.add_argument('--no-logging',
                       dest='use_logging',
                       action='store_false',
                       default=True,
                       help='Disable logging module output')
    
    args = parser.parse_args()
    
    try:
        print_header("CUSTOM SPECTROGRAM GENERATOR")
        
        # Determine input source
        if args.input_dir:
            input_path = args.input_dir
            is_directory = True
            print_status(f"Input directory: {input_path}", "INFO")
        elif args.input_file:
            input_path = args.input_file
            is_directory = False
            print_status(f"Input file: {input_path}", "INFO")
        else:
            # Interactive mode
            print_status("No input specified, entering interactive mode", "INFO")
            input_path, is_directory = prompt_for_input_source()
            if input_path is None:
                print_status("Operation cancelled", "WARNING")
                return
        
        # Validate input
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            print_status(f"Input path not found: {input_path}", "ERROR")
            return
        
        # Determine output directory
        output_dir = determine_output_directory(input_path, is_directory, args.output_dir)
        print_status(f"Output directory: {output_dir}", "INFO")
        
        # Setup parameters
        if args.input_dir or args.input_file:
            # Use command line parameters
            save_mat = not args.no_mat
            save_plot = not args.no_plots
            generator_params = {
                'win_dur': args.win_dur,
                'overlap': args.overlap,
                'window_type': parse_window_type(args.window_type),
                'nfft': args.nfft,
                'win_length': args.win_length,
                'hop_length': args.hop_length,
                'freq_lims': (args.freq_min, args.freq_max),
                'colormap': args.colormap,
                'clim': (args.clim_min, args.clim_max),
                'log_freq': args.log_freq,
                'max_duration': args.max_duration,
                'clip_start': args.clip_start,
                'clip_end': args.clip_end,
                'clip_pad_seconds': args.clip_pad_seconds,
                'backend': args.backend,
                'scaling': args.scaling,
                'quiet': args.quiet,
                'use_logging': args.use_logging,
            }
        else:
            # Interactive parameter selection
            params = prompt_for_parameters()
            save_mat = params['save_mat']
            save_plot = params['save_plot']
            generator_params = {
                'win_dur': params['win_dur'],
                'overlap': params['overlap'],
                'window_type': params['window_type'],
                'nfft': params['nfft'],
                'win_length': params['win_length'],
                'hop_length': params['hop_length'],
                'freq_lims': (params['freq_min'], params['freq_max']),
                'colormap': params['colormap'],
                'clim': (params['clim_min'], params['clim_max']),
                'log_freq': params['log_freq'],
                'max_duration': params['max_duration'],
                'clip_start': params['clip_start'],
                'clip_end': params['clip_end'],
                'clip_pad_seconds': params.get('clip_pad_seconds', None),
                'backend': params['backend'],
                'scaling': params['scaling'],
                'quiet': params['quiet'],
                'use_logging': params['use_logging'],
            }
        
        # Create spectrogram generator
        generator = SpectrogramGenerator(**generator_params)
        
        # Process files
        print_header("PROCESSING")
        start_time = time.time()
        
        summary = process_audio_files(
            input_path, output_dir, is_directory, 
            generator, save_mat, save_plot
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        print_header("RESULTS")
        print_status(f"Total files processed: {summary['total_files']}", "INFO")
        print_status(f"Successful: {summary['successful']}", "SUCCESS")
        if summary['failed'] > 0:
            print_status(f"Failed: {summary['failed']}", "WARNING")
        print_status(f"Processing time: {processing_time:.1f} seconds", "INFO")
        print_status(f"Output saved to: {summary['output_directory']}", "SUCCESS")
        
        # List any errors
        errors = [r for r in summary['results'] if 'error' in r]
        if errors:
            print("\nErrors encountered:")
            for error_result in errors:
                print(f"  {Path(error_result['audio_file']).name}: {error_result['error']}")
        
        print_status("Spectrogram generation completed!", "SUCCESS")
        
    except KeyboardInterrupt:
        print_status("\nOperation interrupted by user", "WARNING")
        return
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        logger.exception("Unexpected error during processing")
        return

if __name__ == "__main__":
    main() 
