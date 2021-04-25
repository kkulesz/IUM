import torch
from torch import nn
from torch.utils.data import DataLoader
from nn.dataset import PurchaseDataset

data_csv_file = 'data/data.csv'
learning_rate = 1e-5
batch_size = 40
epochs = 10


class MatchPredictor(nn.Module):
    def __init__(self):
        super(MatchPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(38, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.stack(x)


def get_training_data():
    return PurchaseDataset(
        train=True,
        data_csv_file=data_csv_file
    )


def get_test_data():
    return PurchaseDataset(
        train=False,
        data_csv_file=data_csv_file
    )


def get_train_dataloader(td):
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def get_test_dataloader(td):
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, sample in enumerate(dataloader):
        X = sample['stats'].float()
        y = sample['success'].type(torch.LongTensor)
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample in dataloader:
            X = sample['stats'].float()
            y = sample['success'].type(torch.LongTensor)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    tr_data = get_training_data()
    te_data = get_test_data()
    train_dl = get_train_dataloader(tr_data)
    test_dl = get_test_dataloader(te_data)

    network = MatchPredictor()
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(network.parameters(), lr=learning_rate)

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_dl, network, lf, opt)
        test_loop(test_dl, network, lf)

    torch.save(network.state_dict(), "model.pth")
