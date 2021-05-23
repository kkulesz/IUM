import pandas as pd
import logging
from nn.model import get_model
from naive_classifier.naive_classifier import NaiveClassifier


class Logic:
    def __init__(self):
        y_column = 'successful'
        data_to_initialize = pd.read_csv('../data_utils/data/data_no_cats.csv')
        y = data_to_initialize[y_column]
        X = data_to_initialize.drop([y_column], axis=1)

        logging.basicConfig(filename='ab_events.log', level=logging.DEBUG)

        self.__dummy_model = NaiveClassifier(X, y)
        self.__correct_model = get_model(X.shape[1])

    def handle_predict_request(self, json_input):
        user_id_col = 'user_id'
        session_id_col = 'session_id'

        user_id = json_input[user_id_col]
        session_id = json_input[session_id_col]

        json_input.pop(user_id_col)
        json_input.pop(session_id_col)

        group = "A" if user_id % 2 == 0 else "B"

        if group == "A":
            print("dummy")
            prediction = self.__dummy_predict(json_input)
        else:
            print("correct")
            prediction = self.__correct_predict(json_input)

        self.__log_prediction(group, session_id, prediction)

        return prediction

    def handle_logging_result(self, json_input):
        session_id = json_input['session_id']
        result = json_input['result']
        self.__log_end_of_session_result(session_id, result)

    def __dummy_predict(self, inputs):
        return self.__dummy_model.dummy_predict(inputs)

    def __correct_predict(self, inputs):
        return self.__correct_model.predict_no_grad(inputs)

    def __log_prediction(self, model, session_id, prediction):
        log_msg = f"Prediction: {model}, {session_id}, {prediction}"
        print(log_msg)
        logging.info(log_msg)

    def __log_end_of_session_result(self, session_id, result):
        logging.info(f"Result: {session_id}, {result}")
