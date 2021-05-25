import torch
from torch import nn


class PurchasePredictor(nn.Module):

    def __init__(self, input_size):
        super(PurchasePredictor, self).__init__()

        hidden_layer_size = int(input_size / 2)

        self.linear1 = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out2 = self.relu(out)
        # out3 = self.linear2(out2)
        return torch.sigmoid(out2)  # Sigmoid zamiast Softmaxa jest do binarnej klasyfikacji

    def predict_no_grad(self, input):  # to pewnie jest bloat i jest juz zaimplementowane, ale zmeczony jestem
        with torch.no_grad():
            X = torch.tensor(list(input.values()))
            prediction = self(X)
            prediction_reshaped = torch.reshape(prediction, (-1,))
            predicted_class = (prediction_reshaped > 0.65)
            return predicted_class.item()


def get_model(input_size):
    model = PurchasePredictor(input_size)
    model.load_state_dict(torch.load("../nn/model.pth"))
    model.eval()
    return model
