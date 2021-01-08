from flask import Flask
import pickle
from flask import request
import numpy as np
import pandas as pd
import os

model_pickle = pickle.load(open("model.p", "rb"))
clf = pickle.loads(model_pickle)

app = Flask(__name__)


@app.route('/predict_single')
def predict_single():
    wealth = request.args.get('wealth')
    religiousness = request.args.get('religiousness')
    if type(wealth) == str:
        wealth = float(wealth)
    if type(religiousness) == str:
        religiousness = float(religiousness)
    X_pred = np.array((wealth, religiousness))
    y_pred = clf.predict(X_pred.reshape(1, -1))
    if (y_pred[0]) == 0:
        return 'Democrat'
    else:
        return 'Republican'


@app.route('/predict_many')
def predict_many():
    X_new_json = request.args.get('data')
    X_pred = pd.read_json(X_new_json)
    y_pred = clf.predict(X_pred)
    y_json = pd.DataFrame(y_pred).to_json()
    return y_json



def main():
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))



if __name__ == '__main__':
    main()
