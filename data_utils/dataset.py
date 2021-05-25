import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

test_size = 0.2


class PurchaseDataset(Dataset):
    def __init__(self, data):
        self.data = data.drop(['user_id'], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, self.data.columns != 'successful'].values, dtype=torch.float32)
        success = torch.tensor(self.data.iloc[idx]['successful'], dtype=torch.float32)
        return {"features": features, "success": success}

    def get_num_of_input_features(self):
        return self.data.shape[1] - 1 # because we also store Y there

    @staticmethod
    def get_test_and_train_datasets(data_file_path):
        data = pd.read_csv(data_file_path)
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True)
        train_dataset = PurchaseDataset(train_data)
        test_dataset = PurchaseDataset(test_data)

        return train_dataset, test_dataset
