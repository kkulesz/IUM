import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils.dataset import PurchaseDataset
from nn.model import PurchasePredictor

# no_cats; le=5e5, batch=10 => max 79.27
# z catsami tez najlepiej 5e5
learning_rate = 1e-5
batch_size = 10
epochs = 30

min_probability_to_success = 0.65


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
    print(f"Test Error: Accuracy: {(100 * correct):>0.3f}%, Avg loss: {test_loss:>8f}")


if __name__ == '__main__':

    # data_path = '../data_utils/data/data.csv'
    data_path = '../data_utils/data/data_no_cats.csv'

    tr_data, te_data = PurchaseDataset.get_test_and_train_datasets(data_path)

    num_of_features = tr_data.get_num_of_input_features()

    train_dl = get_dataloader(tr_data)
    test_dl = get_dataloader(te_data)

    model = PurchasePredictor(num_of_features)
    loss_fun = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        print(f"Epoch {e + 1}: ", end="")
        train_loop(train_dl, model, loss_fun, optimizer)
        test_loop(test_dl, model, loss_fun)

    torch.save(model.state_dict(), "model.pth")
