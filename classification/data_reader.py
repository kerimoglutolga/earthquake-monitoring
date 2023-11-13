import logging 
import os 

import numpy as np
import pandas as pd 

import json
import random 
from collections import defaultdict

import h5py
import obspy
from scipy.interpolate import interp1d
from tqdm import tqdm


def normalize(data, axis=(2,)):
    """Normalize data across specified axes.
    Expected data shape: (batch, channels, timesteps)"""
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1  # To avoid division by zero
    data /= std_data
    return data

def normalize_long(data, axis=2, window=6000):
    """
    Normalize data using a sliding window approach.
    Expected data shape: (batch, channels, timesteps)
    """
    batch, channels, timesteps = data.shape
    if window is None or window > timesteps:
        window = timesteps
    shift = window // 2

    dtype = data.dtype
    std = np.zeros((batch, channels, timesteps))
    mean = np.zeros((batch, channels, timesteps))

    # Apply window-based normalization for each batch and channel
    for b in range(batch):
        for c in range(channels):
            data_pad = np.pad(data[b, c], (window // 2, window // 2), mode="reflect")
            for t in range(0, timesteps, shift):
                w_start, w_end = t, min(t + window, timesteps)
                std[b, c, w_start:w_end] = np.std(data_pad[w_start:w_start + window])
                mean[b, c, w_start:w_end] = np.mean(data_pad[w_start:w_start + window])

    # Interpolation for smooth transition between windows
    for b in range(batch):
        for c in range(channels):
            t_interp = np.arange(timesteps, dtype="int")
            std_interp = interp1d(np.arange(timesteps), std[b, c], kind="slinear")(t_interp)
            mean_interp = interp1d(np.arange(timesteps), mean[b, c], kind="slinear")(t_interp)
            std_interp[std_interp == 0] = 1.0
            data[b, c] -= mean_interp
            data[b, c] /= std_interp

    return data.astype(dtype)


def normalize_batch(data, window=6000):
    """
    Normalize data using a sliding window approach.
    Expected data shape: (batch, channels, timesteps)
    """
    batch, channels, timesteps = data.shape
    if window is None or window > timesteps:
        window = timesteps
    shift = window // 2

    # Padding and Initialization
    data_pad = np.pad(data, ((0, 0), (0, 0), (window // 2, window // 2)), mode="reflect")
    t = np.arange(0, timesteps, shift, dtype="int")
    std = np.zeros([batch, channels, len(t) + 1])
    mean = np.zeros([batch, channels, len(t) + 1])

    # Sliding Window Calculations
    for i in range(1, len(t)):
        std[:, :, i] = np.std(data_pad[:, :, i * shift : i * shift + window], axis=2)
        mean[:, :, i] = np.mean(data_pad[:, :, i * shift : i * shift + window], axis=2)

    t = np.append(t, timesteps)
    std[:, :, -1], mean[:, :, -1] = std[:, :, -2], mean[:, :, -2]
    std[:, :, 0], mean[:, :, 0] = std[:, :, 1], mean[:, :, 1]

    # Interpolation for Smooth Transition
    t_interp = np.arange(timesteps, dtype="int")
    std_interp = np.zeros_like(data)
    mean_interp = np.zeros_like(data)

    for b in range(batch):
        for c in range(channels):
            std_interp[b, c] = interp1d(t, std[b, c], axis=0, kind="slinear")(t_interp)
            mean_interp[b, c] = interp1d(t, mean[b, c], axis=0, kind="slinear")(t_interp)

    std_interp[std_interp == 0] = 1.0

    # Normalization
    normalized_data = (data - mean_interp) / std_interp

    # Handling Sparse Channels
    tmp = np.sum(std_interp, axis=(1, 2))
    nonzero = np.count_nonzero(tmp, axis=-1)

    # Check if nonzero is a scalar and adjust if necessary
    if nonzero.ndim == 0:
        nonzero = np.array([nonzero])

    # Properly broadcast scale_factor over batch dimension
    scale_factor = 3.0 / np.maximum(nonzero[:, np.newaxis, np.newaxis], 1)
    mask = (nonzero > 0)[:, np.newaxis, np.newaxis]
    normalized_data *= np.where(mask, scale_factor, 1)

    return normalized_data


def random_shift(sample, itp, its, shift_range=None):
    # Define helper functions
    flattern = lambda x: np.array([i for trace in x for i in trace], dtype=float)
    shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]

    # Flatten the pick times
    itp_flat = flattern(itp)
    its_flat = flattern(its)
    
    hi = np.round(np.median(itp_flat[~np.isnan(itp_flat)])).astype(int)
    lo = -(sample.shape[1] - np.round(np.median(its_flat[~np.isnan(its_flat)])).astype(int))
    if shift_range is None:
        shift = np.random.randint(low=lo, high=hi + 1)
    else:
        shift = np.random.randint(low=max(lo, shift_range[0]), high=min(hi + 1, shift_range[1]))

    shifted_sample = np.zeros_like(sample)
    if shift > 0:
        shifted_sample[:, :-shift] = sample[:, shift:]
    elif shift < 0:
        shifted_sample[:, -shift:] = sample[:, :shift]
    else:
        shifted_sample = sample

    return shifted_sample, shift_pick(itp, shift), shift_pick(its, shift), shift

# Example usage
# sample = np.random.randn(3, 6000)
# itp, its = [...], [...]  # Define pick times
# shifted_sample, shifted_itp, shifted_its, shift = random_shift(sample, itp, its, ...)


    def stack_events(self, sample_old, itp_old, its_old, shift_range=None, mask_old=None):
        i = np.random.randint(self.num_data)
        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == -1:
            return sample_old, itp_old, its_old

        sample = np.copy(meta["data"])
        itp = meta["itp"]
        its = meta["its"]
        if mask_old is not None:
            mask = np.copy(meta["mask"])
        sample = normalize(sample)
        sample, itp, its, shift = self.random_shift(sample, itp, its, itp_old, its_old, shift_range)

        if shift != 0:
            sample_old += sample
            # itp_old = [np.hstack([i, j]) for i,j in zip(itp_old, itp)]
            # its_old = [np.hstack([i, j]) for i,j in zip(its_old, its)]
            itp_old = [i + j for i, j in zip(itp_old, itp)]
            its_old = [i + j for i, j in zip(its_old, its)]
            if mask_old is not None:
                mask_old = mask_old * mask

        return sample_old, itp_old, its_old, mask_old

    def cut_window(self, sample, target, itp, its, select_range):
        shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
        sample = sample[select_range[0] : select_range[1]]
        target = target[select_range[0] : select_range[1]]
        return (sample, target, shift_pick(itp, select_range[0]), shift_pick(its, select_range[0]))


class DataConfig:
    seed = 123
    use_seed = True
    n_channel = 3
    n_class = 3
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [6000, 1, n_channel]
    Y_shape = [6000, 1, n_class]
    min_event_gap = 3 * sampling_rate
    label_shape = "gaussian"
    label_width = 30
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DataReader:
    def __init__(
        self, data_file, csv_file, config=DataConfig(), sampling_rate=100, highpass_filter=0, 
    ):
        self.buffer = {}
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.dt = config.dt
        self.dtype = config.dtype
        self.label_shape = config.label_shape
        self.label_width = config.label_width
        self.config = config
        self.format = format
        self.highpass_filter = highpass_filter
        self.response = None
        self.sampling_rate = sampling_rate
        
        self.metadata = pd.read_csv(csv_file, low_memory=False)
        self.h5 = h5py.File(data_file, "r", libver="latest", swmr=True)
        self.data_list = list(self.h5_data.keys())
        self.num_data = len(self.data_list)
        

    def __len__(self):
        return self.num_data
    

    def read_hdf5(self, fname):
        data = self.h5_data[fname][()]
        attrs = self.h5_data[fname].attrs
        meta = {}
        if len(data.shape) == 2:
            meta["data"] = data[:, np.newaxis, :]
        else:
            meta["data"] = data
        if "p_idx" in attrs:
            if len(attrs["p_idx"].shape) == 0:
                meta["itp"] = [[attrs["p_idx"]]]
            else:
                meta["itp"] = attrs["p_idx"]
        if "s_idx" in attrs:
            if len(attrs["s_idx"].shape) == 0:
                meta["its"] = [[attrs["s_idx"]]]
            else:
                meta["its"] = attrs["s_idx"]
        if "itp" in attrs:
            if len(attrs["itp"].shape) == 0:
                meta["itp"] = [[attrs["itp"]]]
            else:
                meta["itp"] = attrs["itp"]
        if "its" in attrs:
            if len(attrs["its"].shape) == 0:
                meta["its"] = [[attrs["its"]]]
            else:
                meta["its"] = attrs["its"]
        if "t0" in attrs:
            meta["t0"] = attrs["t0"]
        return meta


    def stack_events(self, sample_old, itp_old, its_old, shift_range=None, mask_old=None):
        i = np.random.randint(self.num_data)
        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == -1:
            return sample_old, itp_old, its_old

        sample = np.copy(meta["data"])
        itp = meta["itp"]
        its = meta["its"]
        if mask_old is not None:
            mask = np.copy(meta["mask"])
        sample = normalize(sample)
        sample, itp, its, shift = self.random_shift(sample, itp, its, itp_old, its_old, shift_range)

        if shift != 0:
            sample_old += sample
            # itp_old = [np.hstack([i, j]) for i,j in zip(itp_old, itp)]
            # its_old = [np.hstack([i, j]) for i,j in zip(its_old, its)]
            itp_old = [i + j for i, j in zip(itp_old, itp)]
            its_old = [i + j for i, j in zip(its_old, its)]
            if mask_old is not None:
                mask_old = mask_old * mask

        return sample_old, itp_old, its_old, mask_old

    def cut_window(self, sample, target, itp, its, select_range):
        shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
        sample = sample[select_range[0] : select_range[1]]
        target = target[select_range[0] : select_range[1]]
        return (sample, target, shift_pick(itp, select_range[0]), shift_pick(its, select_range[0]))


class DataReader_train(DataReader):
    def __init__(self, format="numpy", config=DataConfig(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)

        self.min_event_gap = config.min_event_gap
        self.buffer_channels = {}
        self.shift_range = [-2000 + self.label_width * 2, 1000 - self.label_width * 2]
        self.select_range = [5000, 8000]

    def __getitem__(self, i):
        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == None:
            return (np.zeros(self.X_shape, dtype=self.dtype), np.zeros(self.Y_shape, dtype=self.dtype), base_name)

        sample = np.copy(meta["data"])
        itp_list = meta["itp"]
        its_list = meta["its"]

        sample = normalize(sample)
        if np.random.random() < 0.95:
            sample, itp_list, its_list, _ = self.random_shift(sample, itp_list, its_list, shift_range=self.shift_range)
            sample, itp_list, its_list, _ = self.stack_events(sample, itp_list, its_list, shift_range=self.shift_range)
            target = self.generate_label(sample, [itp_list, its_list])
            sample, target, itp_list, its_list = self.cut_window(sample, target, itp_list, its_list, self.select_range)
        else:
            ## noise
            assert self.X_shape[0] <= min(min(itp_list))
            sample = sample[: self.X_shape[0], ...]
            target = np.zeros(self.Y_shape).astype(self.dtype)
            itp_list = [[]]
            its_list = [[]]

        sample = normalize(sample)
        return (sample.astype(self.dtype), target.astype(self.dtype), base_name)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=True, drop_remainder=True):
        dataset = dataset_map(
            self,
            output_types=(self.dtype, self.dtype, "string"),
            output_shapes=(self.X_shape, self.Y_shape, None),
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_test(DataReader):
    def __init__(self, format="numpy", config=DataConfig(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)

        self.select_range = [5000, 8000]

    def __getitem__(self, i):
        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == -1:
            return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype), base_name)

        sample = np.copy(meta["data"])
        itp_list = meta["itp"]
        its_list = meta["its"]

        # sample, itp_list, its_list, _ = self.random_shift(sample, itp_list, its_list, shift_range=self.shift_range)
        target = self.generate_label(sample, [itp_list, its_list])
        sample, target, itp_list, its_list = self.cut_window(sample, target, itp_list, its_list, self.select_range)

        sample = normalize(sample)
        return (sample, target, base_name, itp_list, its_list)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        dataset = dataset_map(
            self,
            output_types=(self.dtype, self.dtype, "string", "int64", "int64"),
            output_shapes=(self.X_shape, self.Y_shape, None, None, None),
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_pred(DataReader):
    def __init__(self, format="numpy", amplitude=True, config=DataConfig(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)

        self.amplitude = amplitude

    def adjust_missingchannels(self, data):
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert tmp.shape[-1] == data.shape[-1]
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def __getitem__(self, i):
        base_name = self.data_list[i]

        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif (self.format == "mseed") or (self.format == "sac"):
            meta = self.read_mseed(
                os.path.join(self.data_dir, base_name),
                response=self.response,
                sampling_rate=self.sampling_rate,
                highpass_filter=self.highpass_filter,
                return_single_station=True,
            )
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        else:
            raise (f"{self.format} does not support!")

        if "data" in meta:
            raw_amp = meta["data"].copy()
            sample = normalize_long(meta["data"])
        else:
            raw_amp = np.zeros([3000, 1, 3], dtype=np.float32)
            sample = np.zeros([3000, 1, 3], dtype=np.float32)

        if "t0" in meta:
            t0 = meta["t0"]
        else:
            t0 = "1970-01-01T00:00:00.000"

        if "station_id" in meta:
            station_id = meta["station_id"]
        else:
            # station_id = base_name.split("/")[-1].rstrip("*")
            station_id = os.path.basename(base_name).rstrip("*")

        if np.isnan(sample).any() or np.isinf(sample).any():
            logging.warning(f"Data error: Nan or Inf found in {base_name}")
            sample[np.isnan(sample)] = 0
            sample[np.isinf(sample)] = 0

        # sample = self.adjust_missingchannels(sample)

        if self.amplitude:
            return (sample, raw_amp, base_name, t0, station_id)
        else:
            return (sample, base_name, t0, station_id)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        if self.amplitude:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, self.dtype, "string", "string", "string"),
                output_shapes=([None, None, 3], [None, None, 3], None, None, None),
                num_parallel_calls=num_parallel_calls,
                shuffle=shuffle,
            )
        else:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, "string", "string", "string"),
                output_shapes=([None, None, 3], None, None, None),
                num_parallel_calls=num_parallel_calls,
                shuffle=shuffle,
            )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_mseed_array(DataReader):
    def __init__(self, stations, amplitude=True, remove_resp=True, config=DataConfig(), **kwargs):
        super().__init__(format="mseed", config=config, **kwargs)

        # self.stations = pd.read_json(stations)
        with open(stations, "r") as f:
            self.stations = json.load(f)
        print(pd.DataFrame.from_dict(self.stations, orient="index").to_string())

        self.amplitude = amplitude
        self.remove_resp = remove_resp
        self.X_shape = self.get_data_shape()

    def get_data_shape(self):
        fname = os.path.join(self.data_dir, self.data_list[0])
        meta = self.read_mseed_array(fname, self.stations, self.amplitude, self.remove_resp)
        return meta["data"].shape

    def __getitem__(self, i):
        fp = os.path.join(self.data_dir, self.data_list[i])
        # try:
        meta = self.read_mseed_array(fp, self.stations, self.amplitude, self.remove_resp)
        # except Exception as e:
        #     logging.error(f"Failed reading {fp}: {e}")
        #     if self.amplitude:
        #         return (np.zeros(self.X_shape).astype(self.dtype), np.zeros(self.X_shape).astype(self.dtype),
        #             [self.stations.iloc[i]["station"] for i in range(len(self.stations))], ["0" for i in range(len(self.stations))])
        #     else:
        #         return (np.zeros(self.X_shape).astype(self.dtype), ["" for i in range(len(self.stations))],
        #             [self.stations.iloc[i]["station"] for i in range(len(self.stations))])

        sample = np.zeros([len(meta["data"]), *self.X_shape[1:]], dtype=self.dtype)
        sample[:, : meta["data"].shape[1], :, :] = normalize_batch(meta["data"])[:, : self.X_shape[1], :, :]
        if np.isnan(sample).any() or np.isinf(sample).any():
            logging.warning(f"Data error: Nan or Inf found in {fp}")
            sample[np.isnan(sample)] = 0
            sample[np.isinf(sample)] = 0
        t0 = meta["t0"]
        base_name = meta["fname"]
        station_id = meta["station_id"]
        #         base_name = [self.stations.iloc[i]["station"]+"."+t0[i] for i in range(len(self.stations))]
        # base_name = [self.stations.iloc[i]["station"] for i in range(len(self.stations))]

        if self.amplitude:
            raw_amp = np.zeros([len(meta["raw_amp"]), *self.X_shape[1:]], dtype=self.dtype)
            raw_amp[:, : meta["raw_amp"].shape[1], :, :] = meta["raw_amp"][:, : self.X_shape[1], :, :]
            if np.isnan(raw_amp).any() or np.isinf(raw_amp).any():
                logging.warning(f"Data error: Nan or Inf found in {fp}")
                raw_amp[np.isnan(raw_amp)] = 0
                raw_amp[np.isinf(raw_amp)] = 0
            return (sample, raw_amp, base_name, t0, station_id)
        else:
            return (sample, base_name, t0, station_id)

    def dataset(self, num_parallel_calls=1, shuffle=False):
        if self.amplitude:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, self.dtype, "string", "string", "string"),
                output_shapes=([None, *self.X_shape[1:]], [None, *self.X_shape[1:]], None, None, None),
                num_parallel_calls=num_parallel_calls,
            )
        else:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, "string", "string", "string"),
                output_shapes=([None, *self.X_shape[1:]], None, None, None),
                num_parallel_calls=num_parallel_calls,
            )
        dataset = dataset.prefetch(1)
        #         dataset = dataset.prefetch(len(self.stations)*2)
        return dataset


###### test ########


def test_DataReader():
    import os
    import timeit

    import matplotlib.pyplot as plt

    if not os.path.exists("test_figures"):
        os.mkdir("test_figures")

    def plot_sample(sample, fname, label=None):
        plt.clf()
        plt.subplot(211)
        plt.plot(sample[:, 0, -1])
        if label is not None:
            plt.subplot(212)
            plt.plot(label[:, 0, 0])
            plt.plot(label[:, 0, 1])
            plt.plot(label[:, 0, 2])
        plt.savefig(f"test_figures/{fname.decode()}.png")

    def read(data_reader, batch=1):
        start_time = timeit.default_timer()
        if batch is None:
            dataset = data_reader.dataset(shuffle=False)
        else:
            dataset = data_reader.dataset(1, shuffle=False)
        sess = tf.compat.v1.Session()

        print(len(data_reader))
        print("-------", tf.data.Dataset.cardinality(dataset))
        num = 0
        x = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        while True:
            num += 1
            # print(num)
            try:
                out = sess.run(x)
                if len(out) == 2:
                    sample, fname = out[0], out[1]
                    for i in range(len(sample)):
                        plot_sample(sample[i], fname[i])
                else:
                    sample, label, fname = out[0], out[1], out[2]
                    for i in range(len(sample)):
                        plot_sample(sample[i], fname[i], label[i])
            except tf.errors.OutOfRangeError:
                break
                print("End of dataset")
        print("Tensorflow Dataset:\nexecution time = ", timeit.default_timer() - start_time)

    data_reader = DataReader_train(data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_train(format="hdf5", hdf5="test_data/data.h5", group="data")

    read(data_reader)

    data_reader = DataReader_test(data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_test(format="hdf5", hdf5="test_data/data.h5", group="data")

    read(data_reader)

    data_reader = DataReader_pred(format="numpy", data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_pred(
        format="mseed", data_list="test_data/mseed_station.csv", data_dir="test_data/waveforms/"
    )

    read(data_reader)

    data_reader = DataReader_pred(
        format="mseed", amplitude=True, data_list="test_data/mseed_station.csv", data_dir="test_data/waveforms/"
    )

    read(data_reader)

    data_reader = DataReader_mseed_array(
        data_list="test_data/mseed.csv",
        data_dir="test_data/waveforms/",
        stations="test_data/stations.csv",
        remove_resp=False,
    )

    read(data_reader, batch=None)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Generate random data and normalize it
    x = np.random.randn(10, 3, 6000) * 10 + 5
    y = normalize_long(x.copy())

    print(np.mean(x, axis=(1,2)))
    print(np.mean(y, axis=(1,2)))