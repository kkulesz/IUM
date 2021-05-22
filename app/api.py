from flask import Flask, jsonify, request
import torch

from nn.model import get_model


app = Flask(__name__)


def predict_purchase(stats, model):
    with torch.no_grad():
        pred = model(stats)
        return pred[0].argmax(0)


@app.route('/predict', methods=['GET'])
def home():
    data = request.json
    input_size = len(data)
    print(input_size)
    x = torch.tensor(list(data.values()))
    model = get_model(input_size)
    success = predict_purchase(x, model)
    print(float(success))


if __name__ == '__main__':
    app.run(debug=True)
