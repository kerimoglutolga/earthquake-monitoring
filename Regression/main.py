from PhasePicker import PickerNet
from PhasePickerTraining import Picker
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from PhaseNetPicker import PhaseNetPicker
from SWAG import SWAGInference
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

# h5_file = '..//data//chunk3.hdf5'
# dataset = WaveformDataset(csv_file=csv_file, h5_file=h5_file)
if __name__ == '__main__':
    state_dict = torch.load('/Users/noahliniger/Documents/GitHub/earthquake-monitoring/PhaseNetRegressor.pth')
    model = PhaseNetPicker()
    model.load_state_dict(state_dict)
    csv_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.csv', '/Users/noahliniger/Downloads/chunk3/chunk3.csv', '/Users/noahliniger/Downloads/chunk4/chunk4.csv'] #['data/chunk2.csv', 'data/chunk3.csv', 'data/chunk4.csv']#, 'data/chunk5.csv', 'data/chunk6.csv']
    h5_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.hdf5', '/Users/noahliniger/Downloads/chunk3/chunk3.hdf5', '/Users/noahliniger/Downloads/chunk4/chunk4.hdf5'] #['data/chunk2.hdf5', 'data/chunk3.hdf5', 'data/chunk4.hdf5']#, 'data/chunk5.hdf5', 'data/chunk6.hdf5']
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=model)
    print("Creating the data loaders...")
    train_loader, test_loader, valid_loader = picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=32, return_snr=False, input_length=5864)
    print("Created the data loaders...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    model.to(device)
    swag = SWAGInference(model=model, train_loader=train_loader, test_loader=test_loader, bma_samples=30)
    swag.fit_swag(num_training_batches = 50)
    swag.predict_probabilities_swag(num_eval_batches=1)


