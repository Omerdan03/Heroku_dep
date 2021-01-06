import requests
import numpy as np
import json
import pandas as pd
from json import JSONEncoder

BASE_URL = 'http://127.0.0.1:5000/'
one = False

def main():
    if one:
        param = {'wealth': 5, 'religiousness': 3}
        response = requests.get(f'{BASE_URL}/predict_single', params=param)
        print(response.status_code)
        print(response.text)
    else:
        X_new = np.random.randint(low=0, high=10, size=(10, 2)).tolist()
        X_json = pd.DataFrame(X_new).to_json()
        param = {'data': X_json}
        response = requests.get(f'{BASE_URL}/predict_many', params=param)
        y_json = response.text
        y_pred = pd.read_json(y_json)
        print(response.status_code)
        print(y_pred)



if __name__ == '__main__':
    main()