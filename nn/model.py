import torch
from torch import nn


class PurchasePredictor(nn.Module):

    def __init__(self, input_size):
        super(PurchasePredictor, self).__init__()

        hidden_layer_size = int(input_size / 2)  # tu tez bardziej pomyslec
        print(input_size)
        print(hidden_layer_size)

        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out2 = self.relu(out)
        out3 = self.linear2(out2)
        return torch.sigmoid(out3)  # sigmoid zamiast Softmaxa jest do binarnej klasyfikacji


def get_model(input_size):
    model = PurchasePredictor(input_size)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model
