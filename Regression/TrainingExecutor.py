from WaveformDataset import WaveformDataset
from torch.utils.data import ConcatDataset, DataLoader
from PhasePickerTraining import Picker
from PhasePicker import PickerNet
from PhaseNetPicker import PhaseNetPicker
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    csv_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.csv', '/Users/noahliniger/Downloads/chunk3/chunk3.csv',\
         '/Users/noahliniger/Downloads/chunk4/chunk4.csv', '/Users/noahliniger/Downloads/chunk5/chunk5.csv', '/Users/noahliniger/Downloads/chunk6/chunk6.csv'] 
    #['data/chunk2.csv', 'data/chunk3.csv', 'data/chunk4.csv']#, 'data/chunk5.csv', 'data/chunk6.csv']
    h5_files = ['/Users/noahliniger/Downloads/chunk2/chunk2.hdf5', '/Users/noahliniger/Downloads/chunk3/chunk3.hdf5', \
        '/Users/noahliniger/Downloads/chunk4/chunk4.hdf5', '/Users/noahliniger/Downloads/chunk5/chunk5.hdf5', '/Users/noahliniger/Downloads/chunk6/chunk6.hdf5'] 
    #['data/chunk2.hdf5', 'data/chunk3.hdf5', 'data/chunk4.hdf5']#, 'data/chunk5.hdf5', 'data/chunk6.hdf5']
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=PhaseNetPicker())#, model=PickerNet())
    print("Creating the data loaders...")
    picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=128, num_workers=10)
    print("Created the data loaders...")
    # path = '/Users/noahliniger/Documents/GitHub/earthquake-monitoring/Regression/Regressor15EpochsBW.pth'
    # avg_losses_test, avg_losses_train, running_losses = picker.trainModel(num_epochs=30, lr=0.001, resume=True, checkpoint_path=path)
    avg_losses_test, avg_losses_train, running_losses = picker.trainModel(num_epochs=30, lr=0.001, training_seed = 2, resume=False)
    print(avg_losses_test, avg_losses_train)
    plt.plot(running_losses)
    plt.savefig('runnning_losses.png')