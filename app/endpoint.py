from flask import Flask, jsonify, request
import logic

app = Flask(__name__)

ab_logic = logic.Logic()


@app.route('/predict', methods=['GET'])
def home():
    data = request.json
    # print("dane_wejsciowe:\n" + str(data))
    prediction = ab_logic.handle_predict_request(data)

    # xD mnie smieszy
    return "{\n\t " \
           "\"prediction\": \"" + str(prediction) + \
           "\"\n}"


if __name__ == '__main__':
    app.run(debug=True)
