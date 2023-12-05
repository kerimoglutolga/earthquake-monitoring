import re
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

class BedrettoDataset(Dataset):
    def __init__(self, csv_file, h5_file, transform=True, tri_width=50, allow_missing_s=False, width = 6000, slack = 500):
        self.metadata = pd.read_csv(csv_file)
        self.h5_file = h5_file
        self.transform = transform # Apply transformations to data
        self.tri_width = tri_width # Width of triangle shaped label
        self.width = width # Width of wave tensor to be obtained
        self.slack = slack # Slack to be used when cutting the wave tensor
        
        if self.width > max(self.metadata['trace_p_arrival_sample'] - self.metadata['trace_s_arrival_sample']) + 2 * self.slack:
            raise ValueError('Width of wave tensor is larger than the distance between P and S arrival')
        
        pattern = r'bucket(\d+)\$(0|1)'
        self.metadata[['bucket_id', 'idx']] = self.metadata['trace_name'].str.extract(pattern)
        self.metadata['bucket_id'] = self.metadata['bucket_id'].astype(int)
        self.metadata['idx'] = self.metadata['idx'].astype(int)
        
        self.metadata = self.metadata[self.metadata['trace_p_arrival_sample'].notna()]
              
        if not allow_missing_s:
            self.metadata = self.metadata[self.metadata['trace_s_arrival_sample'].notna()]            
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        with h5py.File(self.h5_file, 'r') as dtfl:
            attrs = self.metadata.iloc[idx]
            bucket_id = attrs['bucket_id']
            idx = attrs['idx']
            data = dtfl.get('data/bucket' + str(bucket_id))
            
            wave = np.array(data, dtype=np.float32)[idx]
            
            if np.isnan(attrs['trace_p_arrival_sample']): p_arrival = -1
            else: p_arrival = int(attrs['trace_p_arrival_sample'])
            if np.isnan(attrs['trace_s_arrival_sample']): s_arrival = -1
            else: s_arrival = int(attrs['trace_s_arrival_sample'])
            

            if self.transform:
                wave = self.butterworthFilter(wave)
                wave = self.zeroOneScaling(wave)
            
            #wave += np.random.normal(0, 0.02, wave.shape)
            wave_tensor = torch.from_numpy(wave).to(torch.float32)
            wave_tensor, p_arrival, s_arrival = self.random_cut(wave_tensor, p_arrival, s_arrival)
            
            labels = self.getLabels(p_arrival, s_arrival, self.tri_width)
            labels = labels.to(torch.float32)

        return wave_tensor, labels
    
    def triangle_pick(self, arrival: int, sigma: int):
        range = torch.arange(self.width)
        y1 = -(range - arrival) / sigma + 1
        y1 *= (y1 >= 0) & (y1 <= 1)
        y2 = (range - arrival) / sigma + 1
        y2 *= (y2 >= 0) & (y2 < 1)
        return y1 + y2
    
    def random_cut(self, wave_tensor: torch.tensor, p_arrival: int, s_arrival: int)-> torch.tensor:
        """ Perform a random cut to cut down the length of the tensor to self.width while also containing the P and S arrival
        and update the P and S arrival accordingly """
        
        slack = 500  # Define the slack

        # Ensure the cut tensor will contain both P and S arrivals with the defined slack
        #if p_arrival == -1: start = torch.randint(s_arrival + slack - self.width + 1, s_arrival - slack + 1, (1,))
        #elif s_arrival == -1: start = torch.randint(p_arrival + slack - self.width + 1, p_arrival - slack + 1, (1,))
        #else: start = torch.randint(s_arrival + slack - self.width + 1, p_arrival - slack + 1, (1,))
        start = torch.randint(slack, wave_tensor.shape[1] - self.width - slack, (1,))
        
        # Perform a cut on the tensor with the defined width
        cut_tensor = wave_tensor[0][start : start + self.width]

        # Update the P and S arrivals according to the new cut tensor
        #if p_arrival != -1: p_arrival -= start.item()
        #if s_arrival != -1: s_arrival -= start.item()
        p_arrival -= start.item()
        s_arrival -= start.item()
        
        # Pick isn't in window
        if not (0 <= p_arrival and p_arrival <= self.width): p_arrival = -1
        if not (0 <= s_arrival and s_arrival <= self.width): s_arrival = -1

        return cut_tensor.unsqueeze(0), p_arrival, s_arrival
    
    def getLabels(self, p_arrival: int, s_arrival: int, sigma: int = 50):
        """ Triangle shaped label """
      
        if p_arrival != -1: p = self.triangle_pick(p_arrival, sigma)
        else: p = torch.zeros(self.width)
        if s_arrival != -1: s = self.triangle_pick(s_arrival, sigma)
        else: s = torch.zeros(self.width)
        
        n = torch.clamp(torch.ones(self.width) - p - s, 0)
        
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