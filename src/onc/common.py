import os
from dotenv import load_dotenv
from datetime import datetime, timezone

STATUS_PREFIX = {
    "INFO": "ℹ️ ",
    "SUCCESS": "✅ ",
    "WARNING": "⚠️ ",
    "ERROR": "❌ ",
    "PROGRESS": "🔄 ",
}


def load_config(data_dir_override=None):
    """Load ONC token and data directory from environment/.env."""
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    
    # Resolve default data directory relative to the repository root if running from notebook
    default_data_dir = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    # Check if we are inside 'notebooks' dir, if so, go up one level
    if os.path.basename(os.getcwd()) == 'notebooks':
         default_data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
         
    env_data_dir = os.getenv('DATA_DIR')
    if env_data_dir and env_data_dir.strip() == './data': 
         # If .env explicitly says ./data, resolve it relative to wherever we are, 
         # but let's be smart about notebooks
         if os.path.basename(os.getcwd()) == 'notebooks':
              data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
         else:
              data_dir = os.path.abspath(env_data_dir)
    elif env_data_dir:
         data_dir = env_data_dir
    else:
         data_dir = data_dir_override or default_data_dir

    if not onc_token or onc_token == 'your_onc_api_token_here':
        raise ValueError("Please set your ONC_TOKEN in the .env file")

    return onc_token, data_dir


def print_status(message, level="INFO"):
    """Print a status message with level indicator."""
    prefix = STATUS_PREFIX.get(level, "")
    print(f"{prefix}{message}")


def ensure_timezone_aware(dt_obj, tz=timezone.utc):
    """Convert timezone-naive datetime to timezone-aware datetime."""
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=tz)
    return dt_obj


def format_iso_utc(dt_obj: datetime) -> str:
    """Format datetime as ONC-compatible ISO UTC string."""
    dt_obj = ensure_timezone_aware(dt_obj)
    return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def start_and_end_strings(start_date_object, time_delta):
    """Calculate end date and return formatted start/end strings."""
    start_time_str = format_iso_utc(start_date_object)
    end_object = start_date_object + time_delta
    end_time_str = format_iso_utc(end_object)
    return start_time_str, end_time_str
