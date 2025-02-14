from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

digits = load_digits()
X_train, X_test, y_train, y_test = \
    train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

for i in range(len(X_train)):
    json.dump({'X': X_train[i].tolist(), 'y': int(y_train[i])}, open(f'dataset/train{i}.json', 'w'))
for i in range(len(X_test)):
    json.dump({'X': X_test[i].tolist(), 'y': int(y_test[i])}, open(f'dataset/test{i}.json', 'w'))

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

json.dump({ 'coef': lr.coef_.tolist(), 'inter': lr.intercept_.tolist() }, open('logistic.json', 'w+'))
