import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class PurchaseDataset(Dataset):
    def __init__(self, data_csv_file, train):
        data = pd.read_csv(data_csv_file)
        if train is True:
            self.data = data.iloc[:8000]
        else:
            self.data = data.iloc[8000:]
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stats = torch.tensor(self.data.iloc[idx, self.data.columns != 'successful'].values)
        success = torch.tensor(self.data.iloc[idx]['successful'])
        return {"stats": stats, "success": success}
