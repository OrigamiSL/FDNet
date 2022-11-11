import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset

from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


def _get_time_features(dt, timebed):
    if timebed == 'year':
        return np.stack([
            dt.hour.to_numpy(),  # hour of day
            dt.dayofweek.to_numpy(),  # day of week
            dt.day.to_numpy(),  # day of month
            dt.dayofyear.to_numpy(),  # day of year
            dt.month.to_numpy(),  # month of year
            dt.weekofyear.to_numpy(),  # week of year
        ], axis=1).astype(np.float)
    elif timebed == 'year_min':
        return np.stack([
            dt.minute.to_numpy(),  # minute of hour
            dt.hour.to_numpy(),  # hour of day
            dt.dayofweek.to_numpy(),  # day of week
            dt.day.to_numpy(),  # day of month
            dt.dayofyear.to_numpy(),  # day of year
            dt.month.to_numpy(),  # month of year
            dt.weekofyear.to_numpy(),  # week of year
        ], axis=1).astype(np.float)
    elif timebed == 'hour':
        return np.stack([
            dt.hour.to_numpy()
        ], axis=1).astype(np.float)
    elif timebed == 'day':
        return np.stack([
            dt.dayofyear.to_numpy()  # day of year
        ], axis=1).astype(np.float)
    else:
        print('invalide time embedding')
        exit(-1)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='ETTh1.csv',
                 target='OT', criterion='Standard'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min', 'day']
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.label_len, 12 * 30 * 24 + 4 * 30 * 24 - self.label_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        return seq_x, self.set_bed

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='ETTm1.csv',
                 target='OT', criterion='Standard'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min', 'day']
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.label_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.label_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        return seq_x, self.set_bed

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='year', data_path='ECL.csv',
                 target='MT_320', criterion='Standard'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min', 'day']
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.label_len, len(df_raw) - num_test - self.label_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        return seq_x, self.set_bed

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num
