from WaveformDataset import WaveformDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch
from torch import nn
from typing import Tuple
from torch.optim import Adam
from math import inf
from tqdm import tqdm

class Picker:
    def __init__(self, csv_files: list, h5_files: list, model: torch.nn.Module, seed:int=42) -> None:
        self.csv_files = csv_files
        self.h5_files = h5_files
        self.model = model

        # Check for CUDA availability
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Device:", self.device)
        # self.model.to(self.device)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("Device:", self.device)
        self.model.to(self.device)

        torch.manual_seed(seed)
    
    def createDataLoaders(self, frac_train: float, frac_test: float, batch_size:int = 512,\
         num_workers:int=1, return_snr: bool = True, input_length: int = 5900)\
        -> Tuple[DataLoader, DataLoader, DataLoader]:

        datasets = [WaveformDataset(csv_file=self.csv_files[i], h5_file=self.h5_files[i], return_snr=return_snr,\
            input_length=input_length) for i in range(len(self.csv_files))]
        concatenated_dataset = ConcatDataset(datasets)
        
        train_size = int(frac_train * len(concatenated_dataset))
        test_size = int(frac_test * len(concatenated_dataset))
        valid_size = len(concatenated_dataset) - train_size - test_size

        train_dataset, valid_dataset, test_dataset = random_split(concatenated_dataset, [train_size, valid_size, test_size])

        # Create separate DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        return train_loader, test_loader, valid_loader

    def getLoaders(self):
        return self.train_loader, self.test_loader, self.valid_loader


    def trainModel(self, num_epochs: int, lr:float=0.01, weight_decay:float=1e-6, training_seed:int=10, resume:bool=False, checkpoint_path:str=None):
        current_state = torch.random.get_rng_state()

        # Set a new random seed for training
        torch.manual_seed(training_seed) 

        loss_fn = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize variables to keep track of losses
        avg_losses_train = [inf]
        avg_losses_test = [inf]
        running_train = []

        start_epoch = 0

        # Load checkpoint if resume is True
        if resume and checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            avg_losses_test.append(checkpoint['loss'])

        print("Starting the training process...")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"Starting epoch: {epoch}...")

            self.model.train()
            running_av_train = 0

            for i, (waves, labels) in enumerate(self.train_loader):
                waves, labels = waves.to(self.device), labels.to(self.device)
                outputs = self.model(waves)
                loss = loss_fn(outputs, labels.to(torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_av_train = (running_av_train * i + loss.item())/(i+1)

                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration {i/len(self.train_loader)}, Loss: {running_av_train}")
                    running_train.append(running_av_train)

            print(f"Average Training Loss after Epoch {epoch}: {running_av_train}")
            avg_losses_train.append(running_av_train)

            self.model.eval()
            running_av_test = 0
            print("Starting evaluation on test data...")

            for i, (waves, labels) in enumerate(self.test_loader):
                waves, labels = waves.to(self.device), labels.to(self.device)
                outputs = self.model(waves)
                loss = loss_fn(outputs, labels.to(torch.float32))
                running_av_test = (running_av_test * i + loss.item())/(i+1)
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Iteration {i/len(self.test_loader)}, Loss: {running_av_test}")

            if running_av_test < min(avg_losses_test):
                state = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'avg_losses_train': avg_losses_train,
                    'avg_losses_test': avg_losses_test,
                    'loss': running_av_test
                }
                torch.save(state, f'MultiSwag_seed{training_seed}.pth')

            print(f"Average Test Loss after Epoch {epoch}: {running_av_test}")
            avg_losses_test.append(running_av_test)
        
        torch.random.set_rng_state(current_state)

    # def trainModel(self, num_epochs: int, lr:float=0.01, weight_decay:float=1e-6):

    #     loss_fn = nn.MSELoss()
    #     optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    #     avg_losses_train = [inf]
    #     avg_losses_test = [inf]
    #     running_train = []

    #     print("starting the training process...")

    #     for epoch in range(num_epochs):

    #         print(f"starting epoch: {epoch}...")
            
    #         self.model.train()

    #         running_av_train = 0

    #         for i, (waves, labels) in enumerate(self.train_loader):
    #             #print(i)
    #             waves, labels = waves.to(self.device), labels.to(self.device)
    #             #print(f"moved to GPU: {i}...")
    #             outputs = self.model(waves)
    #             loss = loss_fn(outputs, labels.to(torch.float32))
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             running_av_train = (running_av_train * i + loss.item())/(i+1)

    #             if i%100==0:
    #                 print(f"epoch: {epoch}, iteration {i/len(self.train_loader)}, loss: {running_av_train}")
    #                 running_train.append(running_av_train)
            
    #         print(f"Average Training loss after epoch {epoch}: {running_av_train}")
            
    #         avg_losses_train.append(running_av_train)
            
    #         self.model.eval()
            
    #         running_av_test = 0
            
    #         print("Starting evaluation on test data...")

    #         for i, (waves, labels) in enumerate(self.test_loader):
    #             waves, labels = waves.to(self.device), labels.to(self.device)
    #             outputs = self.model(waves)
    #             loss = loss_fn(outputs, labels.to(torch.float32))
    #             running_av_test = (running_av_test * i + loss.item())/(i+1)
    #             if i%100==0:
    #                 print(f"epoch: {epoch}, iteration {i/len(self.test_loader)}, loss: {running_av_test}")
            
    #         if (running_av_test < min(avg_losses_test)):
    #                 state = {
    #                     'epoch': epoch,
    #                     'model_state': self.model.state_dict(),
    #                     'optimizer_state': optimizer.state_dict(),
    #                     'loss': running_av_test,
    #                 }
    #                 torch.save(state, 'Regression/Regressor15EpochsBW.pth')
            
    #         print(f"Average test loss after epoch {epoch}: {running_av_test}")
            
    #         avg_losses_test.append(running_av_test)
        
    #     return avg_losses_test, avg_losses_train, running_train

        

