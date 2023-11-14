import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

class EQDataset(Dataset):
    def __init__(self, csv_file, h5_file, transform=True, tri_width=50, cutting_length=100):
        self.df = pd.read_csv(csv_file, usecols=['trace_name', 'p_arrival_sample', 's_arrival_sample'])
        self.h5_file = h5_file
        self.transform = transform
        self.tri_width = tri_width
        self.range = torch.arange(6000, dtype=torch.float32)
        self.cutting_length = cutting_length
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        with h5py.File(self.h5_file, 'r') as dtfl:
            evi = self.df['trace_name'].iloc[idx]
            dataset = dtfl.get('data/' + str(evi))
            
            ZWave = np.array(dataset, dtype=np.float32)[:,2]

            if self.transform:
                ZWave = self.butterworthFilter(ZWave)
                ZWave = self.zeroOneScaling(ZWave)
                #ZWave = self.normalize(ZWave)


            wave_tensor = torch.from_numpy(ZWave).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)
            
            labels = self.getLabels(dataset.attrs['p_arrival_sample'], dataset.attrs['s_arrival_sample'], self.tri_width)
            labels = labels.to(torch.float32)
            
            wave_tensor, labels = self.shiftSeries(wave_tensor, labels, cutting_length=self.cutting_length)

        return wave_tensor, labels
    
    def triangle_pick(self, arrival: int, sigma: int):
        y1 = -(self.range - arrival) / sigma + 1
        y1 *= (y1 >= 0) & (y1 <= 1)
        y2 = (self.range - arrival) / sigma + 1
        y2 *= (y2 >= 0) & (y2 < 1)
        return y1 + y2
    
    def getLabels(self, p_arrival: int, s_arrival: int, sigma: int = 50):
        """ Triangle shaped label """
        #if (s_arrival - p_arrival < sigma): 
         #   raise Exception(f"P pick {p_arrival} and S {s_arrival} pick are too close for given triangle width {sigma}.")
        
        p = self.triangle_pick(p_arrival, sigma)
        s = self.triangle_pick(s_arrival, sigma)
        n = torch.clamp(torch.ones(6000) - p - s, 0)
        
        return torch.stack((p,s,n))
    
    def shiftSeries(self, wave_tensor: torch.tensor, labels: torch.tensor, cutting_length : int = 100)\
        -> Tuple[torch.tensor, torch.tensor]:
        """ Randomly shift series by uniform random amount on [0, cutting_length] from front """
        
        shift = np.random.randint(low=0, high=cutting_length)
        new_end = len(wave_tensor[0,:]) - cutting_length + shift

        wave_tensor = wave_tensor[0,shift:new_end]
        labels = labels[:, shift:new_end]

        return wave_tensor, labels

    def zeroOneScaling(self, data: np.array) -> np.array:
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    
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
    
    def normalize(data, axis=(1,)):
        """Normalize data across specified axes.
        Expected data shape: (batch, channels, timesteps)"""
        data -= np.mean(data, axis=axis, keepdims=True)
        std_data = np.std(data, axis=axis, keepdims=True)
        std_data[std_data == 0] = 1  # To avoid division by zero
        data /= std_data
        return data
    
    def random_shift(self, wave_tensor, labels, shift_range=None):
        """
        Shift the wave_tensor and adjust the labels accordingly.
        """
        timesteps = wave_tensor.shape[1]
        shift = np.random.randint(shift_range[0], shift_range[1]) if shift_range else 0

        shifted_wave_tensor = torch.roll(wave_tensor, shifts=shift, dims=1)
        # Ensure that shifted data beyond the original boundaries are zeroed
        if shift > 0:
            shifted_wave_tensor[:, :shift] = 0
        elif shift < 0:
            shifted_wave_tensor[:, shift:] = 0

        # Adjust labels
        adjusted_labels = labels.clone().detach()
        adjusted_labels += shift
        # Ensure labels are within valid range
        adjusted_labels = torch.clamp(adjusted_labels, 0, timesteps - 1)

        return shifted_wave_tensor, adjusted_labels

class NoiseDataset(Dataset):
    def __init__(self, csv_file, h5_file, transform=True, cutting_length = 100):
        self.df = pd.read_csv(csv_file, usecols=['trace_name', 'p_arrival_sample', 's_arrival_sample'])
        self.h5_file = h5_file
        self.transform = transform
        self.cutting_length = cutting_length

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
                #ZWave = self.normalize(ZWave)
                
            wave_tensor = torch.from_numpy(ZWave).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)

            labels = torch.stack((torch.zeros(6000), torch.zeros(6000), torch.ones(6000))) # all noise
        
            wave_tensor, labels = self.shiftSeries(wave_tensor, labels, cutting_length=self.cutting_length)

        return wave_tensor, labels
    
    def shiftSeries(self, wave_tensor: torch.tensor, labels: torch.tensor, cutting_length : int = 100)\
        -> Tuple[torch.tensor, torch.tensor]:
        """ Randomly shift series by uniform random amount on [0, cutting_length] from front """
        
        shift = np.random.randint(low=0, high=cutting_length)
        new_end = len(wave_tensor[0,:]) - cutting_length + shift

        wave_tensor = wave_tensor[0,shift:new_end]
        labels = labels[:, shift:new_end]

        return wave_tensor, labels

    def zeroOneScaling(self, data: np.array) -> np.array:
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    
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
    
    def normalize(data, axis=(1,)):
        """Normalize data across specified axes.
        Expected data shape: (batch, channels, timesteps)"""
        data -= np.mean(data, axis=axis, keepdims=True)
        std_data = np.std(data, axis=axis, keepdims=True)
        std_data[std_data == 0] = 1  # To avoid division by zero
        data /= std_data
        return data




