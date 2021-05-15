import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

data_file_path = '../data_utils/data/data.csv'
test_size = 0.2


class PurchaseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stats = torch.tensor(self.data.iloc[idx, self.data.columns != 'successful'].values)
        success = torch.tensor(self.data.iloc[idx]['successful'])
        return {"stats": stats, "success": success}

    @staticmethod
    def get_test_and_train_datasets():
        data = pd.read_csv(data_file_path)
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True)
        train_dataset = PurchaseDataset(train_data)
        test_dataset = PurchaseDataset(test_data)

        return train_dataset, test_dataset
