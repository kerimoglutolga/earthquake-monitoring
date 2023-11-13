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

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
    # Open the HDF5 file in read mode for each item
        with h5py.File(self.h5_file, 'r') as dtfl:
            evi = self.df['trace_name'].iloc[idx]
            dataset = dtfl.get('data/' + str(evi))
            ZWave = np.array(dataset, dtype=np.float32)[:,2]

            if self.transform:
                ZWave = self.butterworthFilter(ZWave)
                ZWave = self.zeroOneScaling(ZWave)

            wave_tensor = torch.from_numpy(ZWave.copy()).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)
            
            labels = self.getLabels(dataset.attrs['p_arrival_sample'], dataset.attrs['s_arrival_sample'])

            labels, wave_tensor = self.shiftSeries(labels, wave_tensor, cutting_length=100)

        return wave_tensor, labels
    
    def triangle_pick(self, arrival: int, sigma: int, length: int):
        x = torch.arange(length, dtype=torch.float32)
        y1 = -(x - arrival) / sigma + 1
        y1 *= (y1 >= 0) & (y1 <= 1)
        y2 = (x - arrival) / sigma + 1
        y2 *= (y2 >= 0) & (y2 < 1)
        y = y1 + y2
        return y
    
    def getLabels(self, p_arrival: int, s_arrival: int, sigma: int = 50, length: int = 6000):
        """ Triangle shaped label """
        if (s_arrival - p_arrival < sigma): 
            raise Exception(f"P pick {p_arrival} and S {s_arrival} pick are too close for given triangle width {sigma}.")
        
        p = self.triangle_pick(p_arrival, sigma, length)
        s = self.triangle_pick(s_arrival, sigma, length)
        n = torch.ones(length) - p - s
        
        return torch.stack((p,s,n))
    
    
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


