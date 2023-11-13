import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

class WaveformDataset(Dataset):
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
            PWave = np.array(dataset, dtype=np.float32)[:,2]

            if self.transform:
                PWave = self.butterworthFilter(PWave)
                PWave = self.zeroOneScaling(PWave)

            wave_tensor = torch.from_numpy(PWave.copy()).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)

            snr = np.array(dataset.attrs['snr_db'])
            labels = torch.tensor([
                dataset.attrs['p_arrival_sample'],
                dataset.attrs['s_arrival_sample'],
                #dataset.attrs['coda_end_sample'][0][0]
            ], dtype=int)
            torch.round_(labels)
            labels = labels.to(torch.int32)

            labels, wave_tensor = self.shiftSeries(labels, wave_tensor, cutting_length=100)

        return wave_tensor, labels, snr
    
    def shiftSeries(self, labels: torch.tensor, wave_tensor: torch.tensor, cutting_length : int = 100)\
        -> Tuple[torch.tensor, torch.tensor]:
        new_length = len(wave_tensor[0,:]) - cutting_length

        if ((labels[0] > 100) and (labels[1] < new_length)):
            shift = np.random.randint(low=0, high=100)
            labels[0] -= shift
            labels[1] -= shift
            wave_tensor = wave_tensor[0,shift:(new_length + shift)]
        
        #Special cases which should never hit unless cutting length is increased beyond approx 400
        elif ((labels[0] <= 100) and (labels[1] < new_length)):
            wave_tensor = wave_tensor[0,:new_length]
        elif ((labels[0] > 100) and (labels[1] >= new_length)):
            wave_tensor = wave_tensor[0,cutting_length:]
            labels[0] -= cutting_length
            labels[1] -= cutting_length

        else:
            raise Exception(f"The P wave is within cutting length and S wave too: P label {labels[0]}, S label {labels[1]}")

        return labels, wave_tensor

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


