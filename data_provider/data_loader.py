
# from pytorch_lightning import LightningDataModule

import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import data_provider.data_prep

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='./data/six/ETT-small/ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




def StandardScaler(tensor):
    m = tensor.mean(0, keepdim=True)
    s = tensor.std(0, unbiased=False, keepdim=True)
    tensor -= m
    tensor /= s + 1e-8
    return tensor


class LobDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, df, input_width, shift, label_width, stride=1):
        self.input_width = (
            input_width  # input_width: # of time steps that are fed into the models
            # input_width_p1 + input_width_p2
        )
        self.shift = shift  # shift: # of timesteps separating the input and the (final) predictions
        self.label_width = label_width  # label_width: # of time steps in the predictions

        self.window_size = self.input_width + self.shift  # [120,24,24] -> window_size=144
        self.label_start = self.window_size - self.label_width  # [120,24,24] -> label_start=144-24=120

        self.length = df.shape[0]
        self.input_slice = slice(0, self.input_width)
        self.label_slice = slice(self.label_start, None)

        self.mask_slice = None
        if self.shift != self.label_width:
            self.mask_slice = slice(self.input_width, self.label_start)

        self.stride = stride

        # splits = [total[i:i+self.window_size] for i in range(0,self.length - self.window_size + 1,self.stride)]
        # df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        inputs = [df[i : i + self.input_width] for i in range(0, self.length - self.window_size + 1, self.stride)]
        labels = [
            df[i + self.label_start : i + self.window_size] for i in range(0, self.length - self.window_size + 1, self.stride)
        ]

        inputs_tensor = torch.from_numpy(np.concatenate(np.expand_dims(inputs, axis=0), axis=0)).to(dtype=torch.float32)
        labels_tensor = torch.from_numpy(np.concatenate(np.expand_dims(labels, axis=0), axis=0)).to(dtype=torch.float32)
        inputs_tensor = StandardScaler(inputs_tensor)
        labels_tensor = StandardScaler(labels_tensor)

        self.X = inputs_tensor[:, :, :-1]  # mid_price not included
        self.y = labels_tensor[:, :, :-1]
        self.target = labels_tensor[:, :, -1].mean(dim=-1, keepdim=True)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.X[index], self.y[index], self.target[index]  # 


# class LobData(LightningDataModule):
#     def __init__(
#         self,
#         data_dir: str = "./data_new/SB_20210625_20210810",
#         batch_size: int = 32,
#         input_width=120,
#         shift=24,
#         label_width=24,
#         stride=1,
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.input_width = input_width
#
#         self.shift = shift  # shift: # of timesteps separating the input and the (final) predictions
#         self.label_width = label_width  # label_width: # of time steps in the predictions
#
#         self.window_size = self.input_width + self.shift  # [120,24,24] -> window_size=144
#         self.label_start = self.window_size - self.label_width  # [120,24,24] -> label_start=144-24=120
#         self.stride = stride
#
#
#     def setup(self, stage: str):
#         train_df, valid_df, test_df = data_prep.read_parquet(self.data_dir)
#         self.test_ds = LobDataset(test_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)
#         self.train_ds = LobDataset(train_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)
#         self.valid_ds = LobDataset(valid_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)
#
#         # lob_full = MNIST(self.data_dir, train=True)
#         # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
#
#     def train_dataloader(self):
#         return DataLoader(self.train_ds, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.valid_ds, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
