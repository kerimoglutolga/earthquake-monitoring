from WaveformDataset import WaveformDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch
from torch import nn
from typing import Tuple
from torch.optim import Adam
from math import inf
from tqdm import tqdm

class Picker:
    def __init__(self, csv_files: list, h5_files: list, model: torch.nn.Module) -> None:
        self.csv_files = csv_files
        self.h5_files = h5_files
        self.model = model

        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def createDataLoaders(self, frac_train: float, frac_test: float, batch_size:int = 64)\
        -> Tuple[DataLoader, DataLoader, DataLoader]:

        datasets = [WaveformDataset(csv_file=self.csv_files[i], h5_file=self.h5_files[i]) for i in range(len(self.csv_files))]
        concatenated_dataset = ConcatDataset(datasets)
        
        train_size = int(frac_train * len(concatenated_dataset))
        test_size = int(frac_test * len(concatenated_dataset))
        valid_size = len(concatenated_dataset) - train_size - test_size

        train_dataset, valid_dataset, test_dataset = random_split(concatenated_dataset, [train_size, valid_size, test_size])

        # Create separate DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=15)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=15)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def getLoaders(self):
        return self.train_loader, self.test_loader, self.valid_loader

    def trainModel(self, num_epochs: int, lr:float=0.01):

        loss_fn = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        avg_losses_train = [inf]
        avg_losses_test = [inf]
        running_train = []

        for epoch in range(num_epochs):
            
            self.model.train()

            avg_loss_train = 0
            running_av = 0
            for i, (waves, labels, _) in enumerate(self.train_loader):
                waves, labels = waves.to(self.device), labels.to(self.device)
                outputs = self.model(waves.reshape(waves.size(0),1,-1))
                loss = loss_fn(outputs, labels.to(torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_train += loss.item()/len(self.train_loader)
                running_av = (running_av * i + loss.item())/(i+1)
                if i%100==0:
                    running_train.append(running_av)

            avg_losses_train.append(avg_loss_train)
            
            self.model.eval()
            
            avg_loss_test = 0

            for i, (waves, labels, _) in enumerate(self.test_loader):
                waves, labels = waves.to(self.device), labels.to(self.device)
                outputs = self.model(waves.reshape(waves.size(0),1,-1))
                loss = loss_fn(outputs, labels.to(torch.float32))
                avg_loss_test += loss.item()/len(self.test_loader)

            if (avg_loss_test < min(avg_losses_test)):
                torch.save(self.model.state_dict(), 'BestModel.pth')
            
            avg_losses_test.append(avg_loss_test)
        
        return avg_losses_test, avg_loss_train, running_train



