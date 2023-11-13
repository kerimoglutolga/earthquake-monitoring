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
    csv_files = ['data/chunk2.csv']
    h5_files = ['data/chunk2.hdf5']
    picker = Picker(csv_files=csv_files, h5_files=h5_files, model=PickerNet())
    print("Creating the data loaders...")
    picker.createDataLoaders(frac_train=0.7, frac_test=0.15, batch_size=64)
    print("Created the data loaders...")
    avg_losses_test, avg_losses_train, mselosses = picker.trainModel(num_epochs=1)
    print(avg_losses_test, avg_losses_train)
    plt.plot(mselosses)
    plt.savefig('training_mse_loss.png')




    # csv_files = ['data/chunk2.csv']#, '..//data//chunk3.csv', '..//data//chunk4.csv']
    # h5_files = ['data/chunk2.hdf5']#, '..//data//chunk3.hdf5', '..//data//chunk4.hdf5']

    # datasets = [WaveformDataset(csv_file=csv_files[i], h5_file=h5_files[i]) for i in range(len(csv_files))]

    # # Concatenate the datasets
    # concatenated_dataset = ConcatDataset(datasets)

    # # Now you can use the concatenated dataset with a DataLoader
    # print("creating dataloader")
    # dataloader = DataLoader(concatenated_dataset, batch_size=64, shuffle=True, num_workers=4)

    # # all_labels = []

    # # # Iterate over your DataLoader to collect all labels
    # # for _, labels, _ in tqdm(dataloader):
    # #     # Labels are expected to be a batch of tensors, so they need to be concatenated
    # #     all_labels.append(labels)

    # # # Now you have a list of batches, you need to concatenate them into a single tensor
    # # all_labels = torch.cat(all_labels, dim=0)

    # # # Convert to a NumPy array for plotting
    # # all_labels = all_labels.numpy()

    # for series, labels, _ in tqdm(dataloader):
    #     # Labels are expected to be a batch of tensors, so they need to be concatenated
    #     print(torch.min(series), torch.max(series))
    #     break