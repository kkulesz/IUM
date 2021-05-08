from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# https://machinelearningmastery.com/how-to-develop-and-evaluate-naive-classifier-strategies-using-probability/

data = pd.read_csv('../data_utils/data/data.csv')
y = data['successful']
X = data.drop(['successful'], axis=1)

dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X, y)
dummy_model.predict(X)
score = dummy_model.score(X, y)
print(score)
