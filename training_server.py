import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def true_boundary_voting_pred(wealth, religiousness):
    return religiousness - 0.1 * ((wealth - 5) ** 3 - wealth ** 2 + (wealth - 6) ** 2 + 80)


def generate_data(m, seed=None):
    # if seed is not None, this function will always generate the same data
    np.random.seed(seed)

    X = np.random.uniform(low=0.0, high=10.0, size=(m, 2))
    y = np.sign(true_boundary_voting_pred(X[:, 0], X[:, 1]))
    y[y == -1] = 0
    return X, y


def main():
    X, y = generate_data(5000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    model_pickle = pickle.dumps(clf)
    pickle.dump(model_pickle, open("model.p", "wb"))


if __name__ == '__main__':
    main()
