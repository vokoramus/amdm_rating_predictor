import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from datetime import datetime

import requests
import json
from lxml import html

from time import sleep
from random import random

from pprint import pprint

    # for COLAB only
# import os
# from google.colab import drive
# drive.mount('/content/drive')   # mount
# PATH = 'drive/MyDrive/_GB (обучение ИИ)/4.1 Машинное обучение в бизнесе/course_project'
# os.chdir(PATH)
# !ls


# !pip freeze > requirements.txt


def get_response(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'}
    response = requests.get(url, headers=headers)
    dom = html.fromstring(response.text)
    return dom

def get_songs_urls(dom) -> list:
    base = dom.xpath('//tr//a')
    song_name = dom.xpath('//tr//a/text()')
    url_set = dom.xpath('//tr//a/@href')

    return list(zip(song_name, url_set))

songs_urls = list()

def songs_download(url) -> dict:

    db = dict()
    dom = get_response(url)
    songs_urls = get_songs_urls(dom)
    # pprint(songs_urls)

    # TODO: Продумать как поступать с дубликатами

    for i, (song_name, song_url) in enumerate(songs_urls):
        if db.get(song_name):
            song_name = song_name + '_' + str(int(random() * 10**6))

        db[song_name] = song_download(song_name, song_url)

    print(f'Downloaded {len(db)} songs!')

    return db

DELAY = 0.2

def song_download(song_name, song_url) -> dict:
    print(f'Song "{song_name}" have started to download! ... ', end='')
    dom = get_response(song_url)
    chords, song_text, song_view, song_star, song_date = get_song_data(dom)
    sleep(random() * DELAY)
    song_data_dict = {'url': song_url,
                      'chords': chords,
                      'text': song_text,
                      'view': song_view, 
                      'star': song_star, 
                      'date': song_date,
                      }
    print(f'Downloaded!')
    return song_data_dict

def get_song_data(dom) -> tuple:
    chords_set = dom.xpath("//div[@id='song_chords']/img/@alt")
    chords = [s[7:] for s in chords_set]  # убираем начало строки 'Аккорд '

    song_text = dom.xpath("//pre[@itemprop]/text()")
    song_text = clean_text(song_text)

    song_view = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-eye']/parent::*/text()")
    print(song_view, end='')
    song_view = int(song_view[0][1:].replace(',', '_'))

    song_star = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-star']/parent::*/text()")
    song_star = int(song_star[0][1:].replace(',', '_'))

    song_date = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-calendar']/parent::*/text()")
    song_date = song_date[0][1:]

    return chords, song_text, song_view, song_star, song_date

def clean_text(song_text: list) -> list:
    txt2 = []

    # создаем новые строки по переносам строк
    for s in song_text:
        s = s.split('\n')
        txt2 += s

    txt = []
    for s in txt2:
        st = set(s)
        # удаляем пустые строки
        if st - set([' ', '\n']):
            txt.append(s.strip())
    return txt

def get_chords_base(array: np.array) -> set:
    ch_base = set()
    for i in array:
        ch_base = ch_base | set(i)

    return ch_base

# TEST

# def TEST():
#     url = 'https://amdm.ru/akkordi/dana_sokolova/162833/lvinoe_serdce/'
#     dom = get_response(url)

#     w = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-eye']/parent::*/text()")
#     w = int(w[0][1:].replace(',', '_'))
#     q = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-star']/parent::*/text()")
#     q = int(q[0][1:].replace(',', '_'))
#     r = dom.xpath("//div[@class='b-stats']//i[@class='fa fa-calendar']/parent::*/text()")
#     r = r[0][1:]
#     r = datetime.strptime(r, '%d.%m.%Y')

#     print(w, q, r)

# TEST()

x_limit, y_limit = 5, 5

def evaluate_preds(true_values, pred_values, save=False):
    """Оценка качества модели и график preds vs true"""
    
    print("R2:\t" + str(round(r2(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(mse(true_values, pred_values), 3))
         )
    
    plt.figure(figsize=(8,8))
    
    sns.scatterplot(x=pred_values, y=true_values)
    plt.plot([0, x_limit], [0, y_limit], linestyle='--', color='black')  # диагональ, где true_values = pred_values
    
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')
    
    # if save == True:
    #     plt.savefig(REPORTS_FILE_PATH + 'report.png')
    plt.show()

if __name__ == '__main__':

        # указано время скачивания базы песен исполнителя при random задержке = 2 с
    # url = 'https://amdm.ru/akkordi/dana_sokolova/'      # 0m 10sec
    # url = 'https://amdm.ru/akkordi/knyazz/'             # 3m 20sec
    url = 'https://amdm.ru/akkordi/ddt/'                  # Downloaded 620 songs! 10min 44s (sleep = 0.2)


    DB_SOURCE = "saved_db" #@param ["saved_db", "download"]
    SAVE_MODE = "False" #@param ["True", "False"]

    db_filename = './db/ddt.csv'


    # %%time
    if DB_SOURCE == 'download':
        db = songs_download(url)
        raw_df = pd.DataFrame(db).T
        cols = ['chords', 'view', 'star', 'date', 'url', 'text']
        raw_df = raw_df[cols]

        # сохранить скачанный df в файл
        if SAVE_MODE:
            # TODO: задать raw_df.index.name иначе потом считывается как "Undefined: 0" 
            raw_df.to_csv(db_filename, encoding='utf-8')

    elif DB_SOURCE == 'saved_db':
        raw_df = pd.read_csv(db_filename, 
                             encoding='utf-8', 
                            #  index_col=, 
                             verbose=True)

        # затычка
        raw_df = raw_df.rename(columns={'Unnamed: 0': 'song_name'})
        raw_df.set_index('song_name', inplace=True)



    # song_name = 'TEST'
    # song_url = 'https://amdm.ru/akkordi/ddt/9201/glyadi_peshkom/'

    # song_download(song_name, song_url)








