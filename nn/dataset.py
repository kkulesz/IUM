import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class PurchaseDataset(Dataset):
    def __init__(self, data_csv_file, train):
        data = pd.read_csv(data_csv_file)
        if train is True:
            self.data = data.iloc[:7678]
        else:
            self.data = data.iloc[7678:]
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stats = torch.tensor(self.data.loc[:, self.data.columns != 'successful'])
        success = torch.tensor(self.data['successful'])
        return {"stats": stats, "success": success}
