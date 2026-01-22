
import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports if running as script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_download_and_spectrogram():
    """
    Integration test that actually downloads data from ONC and creates a spectrogram.
    WARNING: This consumes API credits/bandwidth and makes real network requests.
    """
    
    # 1. Setup Environment
    load_dotenv(parent_dir / '.env')
    onc_token = os.getenv('ONC_TOKEN')
    
    if not onc_token:
        logger.error("ONC_TOKEN not found in environment variables. Please set it in .env file.")
        sys.exit(1)
        
    # Use a temporary directory or a specific test directory
    test_data_dir = parent_dir / "data" / "test_run"
    if test_data_dir.exists():
        logger.info(f"Cleaning previous test directory: {test_data_dir}")
        shutil.rmtree(test_data_dir)
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using test directory: {test_data_dir}")

    # 2. Initialize Downloader
    downloader = HydrophoneDownloader(onc_token, str(test_data_dir))
    
    # 3. Define Parameters
    # Using a known stable location with audio data
    # Station: Clayoquot Slope (NC89)
    device_code = "ICLISTENHF6324" 
    start_date = "2024-04-01T12:00:00.000Z"
    end_date = "2024-04-01T12:05:00.000Z" # 5 minutes of data
    
    logger.info(f"Starting download for {device_code} from {start_date} to {end_date}")
    
    try:
        # 4. Download Audio
        # Initialize directories for this run
        downloader.setup_directories(
             filetype='flac',
             device_code=device_code,
             download_method='test_run',
             start_date=start_date
        )

        downloader.download_flac_files(
            deviceCode=device_code,
            start_time=start_date,
            end_time=end_date
        )
        
        # Verify Audio Download
        audio_files = list(Path(downloader.audio_path).glob("**/*.flac")) + \
                      list(Path(downloader.audio_path).glob("**/*.wav")) + \
                      list(Path(downloader.audio_path).glob("**/*.mp3"))
                      
        if not audio_files:
            logger.error("No audio files were downloaded!")
            logger.error(f"Checked path: {downloader.audio_path}")
            sys.exit(1)
            
        logger.info(f"Successfully downloaded {len(audio_files)} audio files.")
        for f in audio_files:
            logger.info(f" - {f}")

        # 5. Generate Spectrogram
        logger.info("Initializing SpectrogramGenerator...")
        spec_gen = SpectrogramGenerator(
            win_dur=1.0,
            overlap=0.5,
            freq_lims=(10, 10000),
            clim=(-60, 0),
            use_logging=True
        )
        
        spectrogram_output_dir = test_data_dir / "custom_spectrograms"
        spectrogram_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating spectrograms in {spectrogram_output_dir}...")
        results = spec_gen.process_directory(
            input_dir=downloader.audio_path,
            save_dir=spectrogram_output_dir,
            save_plot=True,
            save_mat=True
        )
        
        # Verify Spectrogram Generation
        generated_pngs = list(spectrogram_output_dir.glob("*.png"))
        generated_mats = list(spectrogram_output_dir.glob("*.mat"))
        
        if len(generated_pngs) == 0:
            logger.error("No PNG spectrograms were generated!")
            sys.exit(1)
            
        if len(generated_mats) == 0:
            logger.error("No MAT files were generated!")
            sys.exit(1)

        logger.info(f"Successfully generated {len(generated_pngs)} PNGs and {len(generated_mats)} MAT files.")
        logger.info(f"Test Run Complete! Output is in {test_data_dir}")
        
    except Exception as e:
        logger.exception("Test failed with exception:")
        sys.exit(1)

if __name__ == "__main__":
    test_real_download_and_spectrogram()
