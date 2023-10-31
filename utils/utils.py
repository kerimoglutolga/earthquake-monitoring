import h5py
import pandas as pd
import numpy as np
import seisbench
import seisbench.data as sbd

def load_dataset_from_disk(metadata_path, data_path, frac=0.01, train_split=0.8, test_split=0.2):
    """Currently only supports STEAD dataset."""

    assert train_split + test_split == 1, "Train and test split should sum up to 1"
    
    # 1. Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[['trace_name', 'p_arrival_sample', 's_arrival_sample']]
    metadata['p_arrival_sample'] = metadata['p_arrival_sample'].astype(int)
    metadata['s_arrival_sample'] = metadata['s_arrival_sample'].astype(int)

    # 2. Randomly sample a fraction of the data
    sample_indices = np.random.choice(metadata.shape[0], int(frac * metadata.shape[0]), replace=False)
    sample_indices.sort()
    sampled_data = []

    with h5py.File(data_path, 'r') as f:
        gdata = f["data"]
        for i in sample_indices:
            row = metadata.loc[i]
            row = row.to_dict()
            waveforms = gdata[row["trace_name"]][()]
            waveforms = waveforms.T  # From WC to CW
            waveforms = waveforms[[2, 1, 0]]  # From ENZ to ZNE

            sampled_data.append(waveforms)
    
    # 3. Create train-test split based on the sampled data
    split_point = int(train_split * len(sampled_data))
    train_data = np.array(sampled_data[:split_point])
    test_data = np.array(sampled_data[split_point:])
    y_train, y_test = metadata.loc[sample_indices[:split_point]].to_numpy(), metadata.loc[sample_indices[split_point:]].to_numpy()     

    return train_data, test_data, y_train, y_test

if __name__ == "__main__":
    train, test, y_train, y_test = load_dataset_from_disk("data/STEAD/chunk2/merged.csv", "data/STEAD/chunk2/merged.hdf5", frac=0.01)
    print(train.shape, y_train.shape)