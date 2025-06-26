import numpy as np
from scipy import signal

class SignalFilter:
    '''
    Class to filter a signal
    Input: Unfiltered signal
    Output: Low,High or Band pass filtered signal
    '''

    def __init__(self, sample_frequency=256, filter_type='low'):
        self.fs = sample_frequency
        self.nyq = 0.5 * self.fs
        self.filter_type = filter_type

    def normalize(self, array):
        """Normalize the input signal/array."""
        return np.apply_along_axis(lambda x: x / np.linalg.norm(x), 0, array)

    def apply_filter(self, *args):
        """Apply the selected filter type to the input arrays."""
        normalized_arrays = [self.normalize(arg) for arg in args]

        if self.filter_type == 'low':
            cutoff = 40.0 / self.nyq  # Set cutoff frequency at 40 Hz
            B, A = signal.butter(4, cutoff, btype='low')
        elif self.filter_type == 'high':
            cutoff = 8.0 / self.nyq  # Set cutoff frequency at 8 Hz
            B, A = signal.butter(4, cutoff, btype='high')
        elif self.filter_type == 'band':
            low = 8 / self.nyq
            high = 40.0 / self.nyq
            B, A = signal.butter(4, [low, high], btype='band')
        else:
            raise ValueError("Invalid filter type. Choose 'low', 'high', or 'band'.")

        return [signal.filtfilt(B, A, n_array) for n_array in normalized_arrays]
