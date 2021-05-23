from flask import Flask, jsonify, request
import logic

app = Flask(__name__)

ab_logic = logic.Logic()


@app.route('/predict', methods=['GET'])
def home():
    data = request.json
    # print("dane_wejsciowe:\n" + str(data))
    prediction = ab_logic.handle_predict_request(data)

    return f"{{\"prediction\": \"{prediction}\"}}"


@app.route('/session_result', methods=['POST'])
def log_result():
    data = request.json
    ab_logic.handle_logging_result(data)
    return "202 OK ;)"


if __name__ == '__main__':
    app.run(debug=True)
