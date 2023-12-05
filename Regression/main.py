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
    def plotPredictions(waves, predictions, labels, min: int = 100, max: int = 5000):

        x = np.linspace(start=min, stop=max, num=max-min)/100
        predictions = predictions/100

        for i in range(32):
            # Creating the primary axis for the wave and vertical lines
            fig, ax1 = plt.subplots(figsize=(20, 6))

            # Plotting the wave
            ax1.plot(x, waves[i,0,min:max], 'k')
            ax1.set_xlabel('Time (s)', fontsize=16)
            ax1.set_ylabel('Density', color='k', fontsize=16)
            ax1.tick_params(axis='both', which='major', labelsize=14)

            #P wave
            map_value = np.mean(predictions[:,i,0])
            stdP = np.std(predictions[:,i,0])
            true_label = labels[i,0]/100

            ax1.vlines(map_value, ymax=1, ymin=0, colors="r", label="Mean Prediction P", linewidth=3) #linestyles="--")
            ax1.vlines(true_label, ymax=1, ymin=0, colors="b", label="True P", linewidth=3) #linestyles="--")

            ax1.vlines(map_value + stdP, ymax=1, ymin=0, colors="gray", label="$\pm$ Standard Deviation", linewidth=3, linestyles="--")
            ax1.vlines(map_value - stdP, ymax=1, ymin=0, colors="gray", linewidth=3, linestyles="--")


            #S wave
            map_value = np.mean(predictions[:,i,1])
            stdS = np.std(predictions[:,i,1])
            true_label = labels[i,1]/100

            ax1.vlines(map_value, ymax=1, ymin=0, colors="g", label="Mean Prediction S",linewidth=3) #linestyles="--")
            ax1.vlines(true_label, ymax=1, ymin=0, colors="orange", label="True S", linewidth=3) #linestyles="--") 

            ax1.vlines(map_value + stdS, ymax=1, ymin=0, colors="pink", linewidth=3, linestyles="--")
            ax1.vlines(map_value - stdS, ymax=1, ymin=0, colors="pink", linewidth=3, linestyles="--")

            ax2 = ax1.twinx()

            ax2.hist(predictions[:,i,0], bins=5, alpha=0.4, color='gray', density=True)
            ax2.hist(predictions[:,i,1], bins=5, alpha=0.4, color='pink', density=True)


            # Adding legend and showing the plot
            ax1.legend(fontsize=14)
            fig.tight_layout()
            plt.savefig(f"/Users/noahliniger/Documents/GitHub/earthquake-monitoring/Regression/Plots/800TrainIter_lr50e-7_50smpl_iter{i}.png")

    checkpoint = torch.load('/Users/noahliniger/Documents/GitHub/earthquake-monitoring/ContinuedTraining.pth')
    model = PhaseNetPicker()
    model.load_state_dict(checkpoint['model_state'])
    csv_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.csv', '/Users/noahliniger/Downloads/chunk3/chunk3.csv', '/Users/noahliniger/Downloads/chunk4/chunk4.csv',\
                    '/Users/noahliniger/Downloads/chunk5/chunk5.csv', '/Users/noahliniger/Downloads/chunk6/chunk6.csv'] 
    h5_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.hdf5', '/Users/noahliniger/Downloads/chunk3/chunk3.hdf5', '/Users/noahliniger/Downloads/chunk4/chunk4.hdf5',\
                    '/Users/noahliniger/Downloads/chunk5/chunk5.hdf5', '/Users/noahliniger/Downloads/chunk6/chunk6.hdf5'] 
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=model)
    print("Creating the data loaders...")
    train_loader, test_loader, valid_loader = picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=32, return_snr=False, input_length=5864)
    print("Created the data loaders...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    swag = SWAGInference(model=model, train_loader=train_loader, test_loader=valid_loader, bma_samples=100, swag_learning_rate=5*10**(-7), deviation_matrix_max_rank=20, swag_update_freq=30)
    swag.fit_swag(num_training_batches = 1000)
    bma_probabilities, per_model_sample_predictions, waves, labels = swag.predict_probabilities_swag(num_eval_batches=2)
    
    p = torch.stack(per_model_sample_predictions).cpu().detach().numpy()
    plotPredictions(min = 0, max = 5000, predictions=p, waves=waves.cpu().detach().numpy(), labels=labels.cpu().detach().numpy())