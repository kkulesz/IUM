from sklearn.dummy import DummyClassifier
import pandas as pd


class NaiveClassifier:
    def __init__(self, X, y):
        self.__dummy_model = DummyClassifier(strategy='most_frequent')
        self.__dummy_model.fit(X, y)

    def get_score(self, X, y):
        return self.__dummy_model.score(X, y)

    def dummy_predict(self, X):
        converted_input = pd.DataFrame(X, index=[0])
        return self.__dummy_model.predict(converted_input)[0] == 1


def read_data():
    data = pd.read_csv('../data_utils/data/data.csv')
    col_to_predict = 'successful'

    y = data[col_to_predict]
    X = data.drop([col_to_predict], axis=1)
    return X, y


if __name__ == '__main__':
    X, y = read_data()
    dummy_model = NaiveClassifier(X, y)
    score = dummy_model.get_score(X, y)
    print(score)  # 0.768
