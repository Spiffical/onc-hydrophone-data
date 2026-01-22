import os


def setup_directories(self, filetype, device_code=None, download_method=None, start_date=None, end_date=None, duration_seconds=None):
    """Setup clean, flat directory structure with optional device, method, and date organization.
    
    New structure (when device_code and download_method provided):
        data/DEVICE/METHOD_DATES/
            onc_spectrograms/   # Downloaded ONC spectrograms (flat, no subdirs)
            audio/              # Downloaded audio files
            custom_spectrograms/ # Generated locally (by generate_spectrograms.py)
    
    Legacy structure (backwards compatibility):
        data/
            onc_spectrograms/   # Downloaded ONC spectrograms
            audio/              # Downloaded audio files
    """
    if device_code and download_method:
        # Create method folder name with date information
        method_folder = self._create_method_folder_name(download_method, start_date, end_date, duration_seconds)
        
        # Clean organized structure: data/DEVICE/METHOD_DATES/
        base_path = os.path.join(self.parent_dir, device_code, method_folder)
        
        # ONC spectrograms - flat structure (no processed/rejects subdirs)
        self.spectrogram_path = os.path.join(base_path, 'onc_spectrograms', '')
        self.input_path = self.spectrogram_path  # Alias for backwards compatibility
        
        # Audio files directory
        self.audio_path = os.path.join(base_path, 'audio', '')
        self.flac_path = self.audio_path  # Alias for backwards compatibility
    else:
        # Legacy structure for backwards compatibility
        self.spectrogram_path = os.path.join(self.parent_dir, 'onc_spectrograms', '')
        self.input_path = self.spectrogram_path
        self.audio_path = os.path.join(self.parent_dir, 'audio', '')
        self.flac_path = self.audio_path
    
    self.onc.outPath = self.spectrogram_path

    # Create all necessary directories (flat structure - no subdirs)
    for folder_path in [self.parent_dir, self.spectrogram_path, self.audio_path]:
        os.makedirs(folder_path, exist_ok=True)


def _create_method_folder_name(self, download_method, start_date=None, end_date=None, duration_seconds=None):
    """Create a descriptive folder name including method and dates"""
    folder_name = download_method
    
    # Add date range information
    if start_date:
        if isinstance(start_date, (list, tuple)):
            # Handle tuple format (year, month, day)
            start_str = f"{start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}"
        elif hasattr(start_date, 'strftime'):
            # Handle datetime object
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = str(start_date)
        
        folder_name += f"_{start_str}"
        
        if end_date:
            if isinstance(end_date, (list, tuple)):
                # Handle tuple format (year, month, day)
                end_str = f"{end_date[0]}-{end_date[1]:02d}-{end_date[2]:02d}"
            elif hasattr(end_date, 'strftime'):
                # Handle datetime object
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)
            
            folder_name += f"_to_{end_str}"
    
    return folder_name
