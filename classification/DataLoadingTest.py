import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

class NoiseDataset(Dataset):
    def __init__(self, csv_file, h5_file, transform=True):
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.h5_file = h5_file
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor, np.array]:
    # Open the HDF5 file in read mode for each item
        with h5py.File(self.h5_file, 'r') as dtfl:
            evi = self.df['trace_name'].iloc[idx]
            dataset = dtfl.get('data/' + str(evi))
            # Only using Z-component
            ZWave = np.array(dataset, dtype=np.float32)[:,2]

            if self.transform:
                ZWave = self.butterworthFilter(ZWave)
                ZWave = self.zeroOneScaling(ZWave)

            wave_tensor = torch.from_numpy(ZWave.copy()).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)

            labels = torch.stack((torch.zeros(6000), torch.zeros(6000), torch.ones(6000)), dtype=torch.int32) # all noise
        
            wave_tensor = self.shiftSeries(labels, wave_tensor, cutting_length=100)

        return wave_tensor, labels
    
    def shiftSeries(self, wave_tensor: torch.tensor, cutting_length : int = 100)\
        -> Tuple[torch.tensor, torch.tensor]:
        new_length = len(wave_tensor[0,:]) - cutting_length

        shift = np.random.randint(low=0, high=cutting_length)
        wave_tensor = wave_tensor[0,shift:(new_length + shift)]

        return wave_tensor

    def zeroOneScaling(self, data: np.array) -> np.array:
        return (data - data.min()) / (data.max() - data.min())
    
    def butterworthFilter(self, data: np.array, lowcut: int = 1, highcut: int = 17, fs: int =100)\
        -> np.array:
        """        
        :param data: Input signal (numpy array).
        :param lowcut: Low frequency cutoff for the Butterworth filter.
        :param highcut: High frequency cutoff for the Butterworth filter.
        :return: Filtered and resampled data.
        """
        # Design the Butterworth filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(N=2, Wn=[low, high], btype='band')

        # Apply the filter
        filtered_data = filtfilt(b, a, data)

        return filtered_data


