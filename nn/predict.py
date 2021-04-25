import torch
from nn.train import PurchasePredictor


def predict_purchase(stats):
    with torch.no_grad():
        pred = model(stats)
        return pred[0].argmax(0)


if __name__ == '__main__':
    model = PurchasePredictor()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    # TODO: getting data to predict if the session will be successful
    # data = get_data()
    # success = predict_purchase(data)
