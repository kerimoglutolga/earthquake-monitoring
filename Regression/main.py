from WaveformDataset import WaveformDataset
from torch.utils.data import ConcatDataset, DataLoader
from PhasePickerTraining import Picker
from PhasePicker import PickerNet
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# h5_file = '..//data//chunk3.hdf5'
# dataset = WaveformDataset(csv_file=csv_file, h5_file=h5_file)
if __name__ == '__main__':
    csv_files = ['data/chunk2.csv', 'data/chunk3.csv', 'data/chunk4.csv', 'data/chunk5.csv', 'data/chunk6.csv']
    h5_files = ['data/chunk2.hdf5', 'data/chunk3.hdf5', 'data/chunk4.hdf5', 'data/chunk5.hdf5', 'data/chunk6.hdf5']
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=PickerNet())
    print("Creating the data loaders...")
    picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=64)
    print("Created the data loaders...")
    avg_losses_test, avg_losses_train, running_losses = picker.trainModel(num_epochs=60, lr=0.01)
    print(avg_losses_test, avg_losses_train)
    plt.plot(running_losses)
    plt.savefig('runnning_losses.png')

