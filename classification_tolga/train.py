import os
import sys 

sys.path.append("./")
sys.path.append("../")

import h5py
import matplotlib.pyplot as plt
import mlflow.pytorch
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seisbench.models as sbm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
	
from models.PhaseNet import PhaseNet, PhaseNetL
from utils.datasets import EQDataset, NoiseDataset

mlflow.set_tracking_uri('http://localhost:5000')  
mlflow.set_experiment('phasenet-training-from-scratch')         
mlflow.pytorch.autolog(log_every_n_step=100)

def get_dataset(data_dir: str, tri_width: int = 50, cutting_length: int = 100):
    datasets = []
    for i in range(1, 7):
        csv_path = os.path.join(data_dir, f"chunk{i}.csv")
        hdf5_path = os.path.join(data_dir, f"chunk{i}.hdf5")
        if os.path.isfile(csv_path) and os.path.isfile(hdf5_path):
            if i == 1:
                datasets.append(NoiseDataset(csv_path, hdf5_path, cutting_length=cutting_length))
            else:
                datasets.append(EQDataset(csv_path, hdf5_path, tri_width=tri_width, cutting_length=cutting_length))

    return ConcatDataset(datasets)


def train_phasenet(data_dir : str = ".\\data\\STEAD", 
                   tri_width=50,
                   cutting_length=100,
                   batch_size=64, 
                   num_workers=10,
                   epochs=100,
                   lr=0.001,
                   
    ):
    dataset = get_dataset(data_dir, tri_width=tri_width, cutting_length=cutting_length)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7,0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = PhaseNetL(lr=lr).to(device)
    
    trainer = pl.Trainer(max_epochs=epochs,)

    with mlflow.start_run(run_name="run-0") as run:
        trainer.fit(model, train_loader, val_loader, ckpt_path="checkpoint.ckpt")
    

if __name__ == "__main__":
    train_phasenet(data_dir=".\\data\\STEAD", tri_width=10, cutting_length=136, batch_size=64, num_workers=0, epochs=2, lr=0.001)