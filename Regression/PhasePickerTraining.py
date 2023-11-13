from WaveformDataset import WaveformDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch
from torch import nn
from typing import Tuple
from torch.optim import Adam
from math import inf

class Trainer:
    def __init__(self, csv_files: list, h5_files: list, model: torch.nn.Module) -> None:
        self.csv_files = csv_files
        self.h5_files = h5_files
        self.model = model
    
    def createDataLoaders(self, frac_train: float, frac_test: float, batch_size:int = 64)\
        -> Tuple[DataLoader, DataLoader, DataLoader]:

        datasets = [WaveformDataset(csv_file=self.csv_files[i], h5_file=self.h5_files[i]) for i in range(len(csv_files))]
        concatenated_dataset = ConcatDataset(datasets)
        
        train_size = int(frac_train * len(concatenated_dataset))
        test_size = int(frac_test * len(concatenated_dataset))
        valid_size = len(concatenated_dataset) - train_size - test_size

        train_dataset, valid_dataset, test_dataset = random_split(concatenated_dataset, [train_size, valid_size, test_size])

        # Create separate DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, valid_loader, test_loader

    def trainModel(self, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader):

        loss_fn = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=0.001)

        avg_losses_train = [inf]
        avg_losses_test = [inf]

        for epoch in range(num_epochs):
            
            self.model.train()

            avg_loss_train = 0
            for i, (waves, labels, _) in enumerate(train_loader):
                outputs = self.model(waves.reshape(waves.size(0),1,-1))
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_train += loss.item()/len(train_loader)

            avg_losses_train.append(avg_loss_train)
            
            self.model.eval()
            
            avg_loss_test = 0
            for i, (waves, labels, _) in enumerate(test_loader):
                outputs = self.model(waves.reshape(waves.size(0),1,-1))
                loss = loss_fn(outputs, labels)
                avg_loss_test += loss.item()/len(test_loader)

            if (avg_loss_test < min(avg_losses_test)):
                torch.save(self.model.state_dict(), 'BestModel.pth')
            
            avg_losses_test.append(avg_loss_test)











# Example proportions: 70% training, 15% validation, 15% testing
train_size = int(0.7 * len(concatenated_dataset))
valid_size = test_size = (len(concatenated_dataset) - train_size) // 2

# Ensure the sum of sizes equals the total size of concatenated dataset
test_size = len(concatenated_dataset) - train_size - valid_size

# Randomly split the dataset
train_dataset, valid_dataset, test_dataset = random_split(concatenated_dataset, [train_size, valid_size, test_size])

# Create separate DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

csv_files = ['data/chunk2.csv']#, '..//data//chunk3.csv', '..//data//chunk4.csv']
h5_files = ['data/chunk2.hdf5']#, '..//data//chunk3.hdf5', '..//data//chunk4.hdf5']

datasets = [WaveformDataset(csv_file=csv_files[i], h5_file=h5_files[i]) for i in range(len(csv_files))]

# Concatenate the datasets
concatenated_dataset = ConcatDataset(datasets)

# Now you can use the concatenated dataset with a DataLoader
print("creating dataloader")
dataloader = DataLoader(concatenated_dataset, batch_size=64, shuffle=True, num_workers=4)

for signals, labels, snrs in dataloader: break

signals = signals.reshape(64,1,-1)

net = PickerNet()

net(signals)