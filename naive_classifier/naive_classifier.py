from sklearn.dummy import DummyClassifier
import pandas as pd


def read_data():
    col_to_predict = 'successful'

    data = pd.read_csv('../data_utils/data/data.csv')
    y = data[col_to_predict]
    X = data.drop([col_to_predict], axis=1)
    return X, y


def get_dummy_score(X, y):
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X, y)
    # dummy_model.predict(X) [0,0,0.....]
    return dummy_model.score(X, y)


if __name__ == '__main__':
    X, y = read_data()
    score = get_dummy_score(X, y)
    print(score)  # 0.768
