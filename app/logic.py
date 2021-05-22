import pandas as pd
from nn.model import get_model
from naive_classifier.naive_classifier import NaiveClassifier


class Logic:
    def __init__(self):
        y_column = 'successful'
        data_to_initialize = pd.read_csv('../data_utils/data/data_no_cats.csv')
        y = data_to_initialize[y_column]
        X = data_to_initialize.drop([y_column], axis=1)

        self.__dummy_model = NaiveClassifier(X, y)
        self.__correct_model = get_model(X.shape[1])

    def handle_predict_request(self, json_input):
        user_id_col = 'user_id'
        # TODO: pewnie trzeba jeszcze session_id i dodac nowy endpoint,
        #  ktory zapisze nam czy sesja zostala zakonczona sukcesem, zeby potem sprawdzic czy sie udalo
        user_id = json_input[user_id_col]
        json_input.pop(user_id_col)

        group = user_id % 2 == 0
        if group:
            print("dummy")
            prediction = self.dummy_predict(json_input)
        else:
            print("correct")
            prediction = self.correct_predict(json_input)

        self.log(group, json_input, prediction)

        return prediction

    def dummy_predict(self, input):
        return self.__dummy_model.dummy_predict(input)

    def correct_predict(self, input):
        return self.__correct_model.predict_no_grad(input)

    def log(self, model, input, prediction):
        pass
