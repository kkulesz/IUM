import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils.dataset import PurchaseDataset

learning_rate = 1e-4
batch_size = 40
epochs = 30

min_probability_to_success = 0.65


class PurchasePredictor(nn.Module):
    def __init__(self, input_size):
        super(PurchasePredictor, self).__init__()

        input_size = input_size - 1  # TODO: nie wiem dlaczego tak?
        hidden_layer_size = int(input_size / 2)  # tu tez bardziej pomyslec

        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out2 = self.relu(out)
        out3 = self.linear2(out2)
        return torch.sigmoid(out3)  # sigmoid zamiast Softmaxa jest do binarnej klasyfikacji


def get_dataloader(td):
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, sample in enumerate(dataloader):
        X = sample['features']
        y = sample['success']
        prediction = model(X)
        prediction = torch.reshape(prediction, (-1,))  # TODO: pozbyc sie tego pozniej

        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()  # update
        optimizer.zero_grad()

        # if batch % 25 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample in dataloader:
            X = sample['features']
            y = sample['success']
            prediction = model(X)
            prediction_reshaped = torch.reshape(prediction, (-1,))

            test_loss += loss_fn(prediction_reshaped, y).item()

            predicted_class = (prediction_reshaped > min_probability_to_success).float()

            correct += float(sum(predicted_class == y))

    test_loss /= size
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f}")


if __name__ == '__main__':
    tr_data, te_data = PurchaseDataset.get_test_and_train_datasets()

    num_of_features = tr_data.get_num_of_input_features()
    print(num_of_features)

    train_dl = get_dataloader(tr_data)
    test_dl = get_dataloader(te_data)

    network = PurchasePredictor(num_of_features)
    loss_fun = nn.BCELoss()
    opt = torch.optim.SGD(network.parameters(), lr=learning_rate)

    for e in range(epochs):
        # print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_dl, network, loss_fun, opt)
        test_loop(test_dl, network, loss_fun)

    # torch.save(network.state_dict(), "model.pth")
