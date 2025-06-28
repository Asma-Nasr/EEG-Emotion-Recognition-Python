import os
import numpy as np
from python_speech_features import mfcc
from src.load import extract_columns


def extract_and_save_mfcc_from_folder(input_folder, output_folder, fs=250, nc=12, p=16, n=128, inc=64):
    """
    Process all .npy files in the input folder, extract MFCC features, and save them to the output folder.

    Parameters:
    - input_folder: str - Path to the folder containing input .npy files.
    - output_folder: str - Path to the folder where output .npy files will be saved.
    - fs: int - Sampling frequency.
    - nc: int - Number of cepstral coefficients.
    - p: int - Frame length in milliseconds.
    - n: int - FFT size.
    - inc: int - Frame step in milliseconds.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all .npy files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            # Load the signal from the .npy file
            file_path = os.path.join(input_folder, filename)
            signals = np.load(file_path)
            signals = extract_columns(signals)

            # Calculate MFCC features for each signal
            mfcc_features = [
                mfcc(signal, fs, numcep=nc, winlen=p/fs, winstep=inc/fs, nfft=n) 
                for signal in signals
            ]

            # Concatenate the MFCC features
            calm_mfcc = np.concatenate(mfcc_features, axis=1)

            # Extract the base name and create new filename
            base_name = filename.split('_')[-1].replace('.npy', '')  # Get the last part before .npy
            new_filename = f"{base_name}_mfcc.npy"  # Format: calm_mfcc.npy
            output_path = os.path.join(output_folder, new_filename)

            # Save the concatenated MFCC features to a new .npy file
            np.save(output_path, calm_mfcc)
