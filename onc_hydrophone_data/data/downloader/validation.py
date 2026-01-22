import glob
import os
import time

import numpy as np
import scipy.io
from PIL import Image

from ..segment import segment2


def check_for_anomalies(self, file_path, anomaly_log=None):
    """Check a spectrogram file for anomalies (black/white bands).
    
    In the flat directory structure, anomalous files are NOT moved - they are just
    logged for reference. This allows users to review and decide what to do.
    
    Args:
        file_path: Path to the spectrogram file
        anomaly_log: Optional list to append anomaly info to
        
    Returns:
        dict with 'has_anomaly', 'file', and 'issues' keys
    """
    result = {'has_anomaly': False, 'file': file_path, 'issues': []}
    
    try:
        image_obj = None
        if file_path.lower().endswith('.png'):
            image_obj = Image.open(file_path)
            image_obj = np.transpose(image_obj, [1, 0, 2])
        elif file_path.lower().endswith('.mat'):
            mat_data = scipy.io.loadmat(file_path)
            if 'SpectData' in mat_data:
                image_obj = mat_data['SpectData']['PSD'][0,0]
            else:
                result['has_anomaly'] = True
                result['issues'].append('No "SpectData" key found in .mat file')
                self.logger.warning(f'No SpectData in {os.path.basename(file_path)}')
                if anomaly_log is not None:
                    anomaly_log.append(result)
                return result

        if image_obj is not None:
            s = np.zeros([np.shape(image_obj)[0], 1])
            anom_indices_black = []
            anom_indices_white = []
            
            for ii in np.arange(0, np.shape(image_obj)[0]):
                s[ii] = np.sum(image_obj[ii])
                if s[ii] < 500:
                    anom_indices_black.append(ii)
                elif s[ii] > 568000:
                    anom_indices_white.append(ii)

            if anom_indices_black:
                seg = segment2(anom_indices_black)
                issue = f"{seg.shape[0]} black segment(s) at [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                result['issues'].append(issue)
                result['has_anomaly'] = True
                self.logger.info(f'Anomaly in {os.path.basename(file_path)}: {issue}')

            if anom_indices_white:
                seg = segment2(anom_indices_white)
                issue = f"{seg.shape[0]} white segment(s) at [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                result['issues'].append(issue)
                result['has_anomaly'] = True
                self.logger.info(f'Anomaly in {os.path.basename(file_path)}: {issue}')

    except Exception as e:
        err_msg = str(e)
        # Truncated files are kept for inspection
        if 'truncated' in err_msg.lower():
            self.logger.warning(f'Truncated file detected: {os.path.basename(file_path)}')
            result['issues'].append(f'Truncated file: {err_msg}')
        else:
            self.logger.warning(f'Error checking {os.path.basename(file_path)}: {err_msg}')
            result['issues'].append(f'Error: {err_msg}')
        result['has_anomaly'] = True
    
    if anomaly_log is not None and result['has_anomaly']:
        anomaly_log.append(result)
    
    return result


def validate_spectrograms(self, filetype='mat'):
    """Validate downloaded spectrograms and generate an anomaly report.
    
    This method scans spectrogram files for anomalies (black/white bands, 
    missing data) and writes a report file. Files are NOT moved - they stay
    in the flat onc_spectrograms/ directory.
    
    Args:
        filetype: File extension to process ('mat' or 'png')
        
    Returns:
        dict with validation summary
    """
    process_start = time.time()
    self.logger.info(f"Validating {filetype} spectrograms in {self.spectrogram_path}")
    
    anomaly_log = []
    
    if filetype == 'mat':
        file_paths = glob.glob(os.path.join(self.spectrogram_path, '*.mat'))
    elif filetype == 'png':
        file_paths = glob.glob(os.path.join(self.spectrogram_path, '*.png'))
    else:
        file_paths = []
    
    self.logger.info(f"Found {len(file_paths)} {filetype.upper()} files to validate")
    
    for file_path in file_paths:
        self.check_for_anomalies(file_path, anomaly_log)
    
    # Write anomaly report if any issues found
    if anomaly_log:
        report_path = os.path.join(self.spectrogram_path, 'anomaly_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Anomaly Report - {len(anomaly_log)} files with issues\n")
            f.write("=" * 60 + "\n\n")
            for entry in anomaly_log:
                f.write(f"File: {os.path.basename(entry['file'])}\n")
                for issue in entry['issues']:
                    f.write(f"  - {issue}\n")
                f.write("\n")
        self.logger.info(f"Anomaly report written to {report_path}")
    
    elapsed = time.time() - process_start
    self.logger.info(f"Validation completed in {elapsed:.2f}s ({len(anomaly_log)} anomalies found)")
    
    return {
        'total_files': len(file_paths),
        'anomalies': len(anomaly_log),
        'anomaly_details': anomaly_log,
        'elapsed_seconds': elapsed,
    }


def process_spectrograms(self, filetype='png'):
    """Legacy method - now just calls validate_spectrograms.
    
    Note: This method previously moved files between subdirectories.
    The new flat structure no longer moves files.
    """
    return self.validate_spectrograms(filetype)
