from flask import Flask, jsonify, request
import logic

app = Flask(__name__)

ab_logic = logic.Logic()


@app.route('/predict', methods=['GET'])
def get_prediction():
    data = request.get_json(force=True)
    prediction = ab_logic.handle_predict_request(data)

    return f"{{\"prediction\": \"{prediction}\"}}"


@app.route('/session_result', methods=['POST'])
def log_result():
    data = request.get_json(force=True)
    ab_logic.handle_logging_result(data)

    return f"{{ \"response\": \"202 OK ;-)\" }}"


if __name__ == '__main__':
    app.run(debug=True)
