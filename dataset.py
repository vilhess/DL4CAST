import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class TSDataset(Dataset):
    def __init__(self, path, seq_len=100, target_len=100, mode="train", univariate=False, target='OT', use_time_features=False):

        self.seq_len = seq_len
        self.target_len = target_len
        self.mode = mode
        self.use_time_features = use_time_features

        self.data = pd.read_csv(path)

        df_stamp = pd.DatetimeIndex(self.data['date'])
        self.data.drop(columns=['date'], inplace=True)

        if univariate:
            self.data = self.data[[target]]

        train_size = int(len(self.data) * 0.7)
        test_size = int(len(self.data) * 0.2)
        val_size = len(self.data) - train_size - test_size

        train = self.data[:train_size].values

        scaler = StandardScaler()
        scaler.fit(train)

        self.data = scaler.transform(self.data.values)

        if self.mode == "train":
            self.data = self.data[:train_size]
            if self.use_time_features:
                self.time_char = time_features(df_stamp[:train_size])
                
        elif self.mode == "val":
            self.data = self.data[train_size-seq_len:train_size + val_size]
            if self.use_time_features:
                self.time_char = time_features(df_stamp[train_size-seq_len:train_size + val_size])

        else:
            self.data = self.data[train_size + val_size  - seq_len:]
            if self.use_time_features:
                self.time_char = time_features(df_stamp[train_size + val_size - seq_len:])

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len + 1
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.target_len]

        if self.use_time_features:
            time_char = self.time_char[index:index + self.seq_len]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(time_char, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        

class TimeFeature:
    def __init__(self):
        pass
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23. - 0.5
    
class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6. - 0.5
    
class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30. - 0.5

class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365. - 0.5

def time_features(dates):
    return np.vstack([feat(dates) for feat in [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]]).T