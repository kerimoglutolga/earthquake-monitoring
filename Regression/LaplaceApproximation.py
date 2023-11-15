from PhasePicker import PickerNet
from PhasePickerTraining import Picker
import matplotlib.pyplot as plt
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import numpy as np
import torch
from laplace import Laplace, marglik_training
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import h5py

if __name__ == '__main__':
    state_dict = torch.load('/Users/noahliniger/Documents/GitHub/earthquake-monitoring/3Epochs_noBW.pth')
    model = PickerNet()
    model.load_state_dict(state_dict)
    csv_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.csv', '/Users/noahliniger/Downloads/chunk3/chunk3.csv', '/Users/noahliniger/Downloads/chunk4/chunk4.csv'] #['data/chunk2.csv', 'data/chunk3.csv', 'data/chunk4.csv']#, 'data/chunk5.csv', 'data/chunk6.csv']
    h5_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.hdf5', '/Users/noahliniger/Downloads/chunk3/chunk3.hdf5', '/Users/noahliniger/Downloads/chunk4/chunk4.hdf5'] #['data/chunk2.hdf5', 'data/chunk3.hdf5', 'data/chunk4.hdf5']#, 'data/chunk5.hdf5', 'data/chunk6.hdf5']
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=PickerNet())
    print("Creating the data loaders...")
    picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=32, return_snr=False)
    print("Created the data loaders...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    model.to(device)
    la = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
    la.fit(picker.valid_loader)
    la.optimize_prior_precision(method='marglik')