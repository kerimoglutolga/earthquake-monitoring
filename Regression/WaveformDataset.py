import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Tuple

class WaveformDataset(Dataset):
    def __init__(self, csv_file, h5_file, transform=True, return_snr: bool = 1, input_length: int = 5900):
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.h5_file = h5_file
        self.transform = transform
        self.return_snr = return_snr
        self.input_length = input_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor, np.array]:
    # Open the HDF5 file in read mode for each item
        with h5py.File(self.h5_file, 'r') as dtfl:
            evi = self.df['trace_name'].iloc[idx]
            dataset = dtfl.get('data/' + str(evi))
            wave = np.array(dataset, dtype=np.float32)[:,2]

            if self.transform:
                #wave = self.butterworthFilter(wave)
                wave = self.zeroOneScaling(wave)

            wave_tensor = torch.from_numpy(wave.copy()).reshape(1,-1).to(torch.float32)  # Shape becomes (1, 6000)

            snr = np.array(dataset.attrs['snr_db'])
            labels = torch.tensor([
                dataset.attrs['p_arrival_sample'],
                dataset.attrs['s_arrival_sample']
                #dataset.attrs['coda_end_sample'][0][0]
            ])
            torch.round_(labels)
            labels = labels.to(torch.float32)

            labels, wave_tensor = self.shiftSeries(labels, wave_tensor, cutting_length=100)
            wave_tensor.unsqueeze_(0)
            wave_tensor = wave_tensor[:,:self.input_length]

        if self.return_snr: return wave_tensor, labels, snr
        else: return wave_tensor, labels
    
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
    
    def normalize(data, axis=(2,)):
        """Normalize data across specified axes.
        Expected data shape: (batch, channels, timesteps)"""
        data -= np.mean(data, axis=axis, keepdims=True)
        std_data = np.std(data, axis=axis, keepdims=True)
        std_data[std_data == 0] = 1  # To avoid division by zero
        data /= std_data
        return data

    def normalize_long(data, axis=2, window=6000):
        """
        Normalize data using a sliding window approach.
        Expected data shape: (batch, channels, timesteps)
        """
        batch, channels, timesteps = data.shape
        if window is None or window > timesteps:
            window = timesteps
        shift = window // 2

        dtype = data.dtype
        std = np.zeros((batch, channels, timesteps))
        mean = np.zeros((batch, channels, timesteps))

        # Apply window-based normalization for each batch and channel
        for b in range(batch):
            for c in range(channels):
                data_pad = np.pad(data[b, c], (window // 2, window // 2), mode="reflect")
                for t in range(0, timesteps, shift):
                    w_start, w_end = t, min(t + window, timesteps)
                    std[b, c, w_start:w_end] = np.std(data_pad[w_start:w_start + window])
                    mean[b, c, w_start:w_end] = np.mean(data_pad[w_start:w_start + window])

        # Interpolation for smooth transition between windows
        for b in range(batch):
            for c in range(channels):
                t_interp = np.arange(timesteps, dtype="int")
                std_interp = interp1d(np.arange(timesteps), std[b, c], kind="slinear")(t_interp)
                mean_interp = interp1d(np.arange(timesteps), mean[b, c], kind="slinear")(t_interp)
                std_interp[std_interp == 0] = 1.0
                data[b, c] -= mean_interp
                data[b, c] /= std_interp

        return data.astype(dtype)

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



