import requests
import urllib.request
import json 
import pandas as pd

from flask import jsonify

# загрузить csv из файла
X_train = pd.read_csv("./db/X_train_p.csv")
y_train = pd.read_csv("./db/y_train_p.csv")

X_test = pd.read_csv("./db/X_test_p.csv")
y_test = pd.read_csv("./db/y_test_p.csv")

# Преобразуем столбец 'chords'
X_train['chords'] = X_train['chords'].apply(lambda x: eval(x))
X_test['chords'] = X_test['chords'].apply(lambda x: eval(x))


# формируем запрос
def send_json(x):
    body = x.to_dict()
    myurl = 'http://4f85-35-196-4-248.ngrok.io' + '/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}

    response = requests.post(myurl, json=body, headers=headers)
    print(f'response = {response}')
    # return
    return response.json()['predictions']


# предсказание для одной песни
idx = 0
sample_song = X_train.iloc[idx]  # DF из одной песни

print(sample_song)

response = send_json(sample_song)
print(response)


# Масса запросов

N = 40

# %%time
predictions = X_train.iloc[:N].apply(lambda x: send_json(x), axis=1)

print(predictions.values[:5])
