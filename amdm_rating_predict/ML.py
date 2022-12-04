# Machine Learning (PIPELINES)
import numpy as np
import pandas as pd
import dill

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        # print(__class__, 'fit OK')
        return self

    def transform(self, X):
        # print(__class__, 'transform OK')
        return X[self.key]


class ChordFeatureToListMaker(BaseEstimator, TransformerMixin):
    """
    Transformer to make a List type to Chord column
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # print(__class__, 'fit OK')
        return self
    
    def transform(self, X):
        # print(__class__, 'transform START', f"X['chords'].dtypes = {X['chords'].dtypes} \n")
        X['chords'] = X['chords'].apply(lambda x: eval(x))
        # print(__class__, 'transform OK')
        return X


class NoChordsSongsDeletor(BaseEstimator, TransformerMixin):
    """
    Transformer to delete songs with no chords
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # print(__class__, 'fit OK')
        return self
    
    def transform(self, X_):
        # print(__class__, 'transform START')
        X = X_.copy()  # иначе pipeline будет одноразовый (будет портить DataFrame inplace)

        # удалить песни без аккордов (обработка строк)
        mask = X['chords'].apply(lambda x: x == [])
        X.drop(X[mask].index, inplace=True)
        # df[df['column name'].map(len) < 2]  # механизм применения к каждому элементу столбца

        # print(__class__, 'transform OK')
        return X


class ChordsFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Chords new features creating
    """
    def __init__(self):
        self.chords_base = set()
        self.chords_columns = []

    def fit(self, X, y=None):
        # print(__class__, 'fit START')
        X.apply(lambda x: self.type_assert(x), axis=1)

        # получаем список используемых аккордов во всех песнях (для создания фичей)
        self.get_chords_base(X['chords'])
        # print(f'Кол_во уникальных аккордов: {len(chords_base)}')

        # chords_columns list
        FIRST_CHORD_COLUMN_No = 3
        self.chords_columns = X.columns[FIRST_CHORD_COLUMN_No: FIRST_CHORD_COLUMN_No + len(self.chords_base)]
        # print(len(chords_columns), (len(chords_columns) == len(chords_base)), chords_columns, sep='\n')
        # print(__class__, 'fit OK')
        return self

    def type_assert(self, x):
        assert type(x['chords']) == list, f"  Bad data type for: {x.name},\n  type={type(x['chords'])}, \n  x['chords'] = {x['chords']}"
        return

    def get_chords_base(self, array: np.array):
        for i in array:
            self.chords_base = self.chords_base | set(i)

    def transform(self, X_):
        # print(__class__, 'transform START')
        X = X_.copy()  # иначе pipeline будет одноразовый (будет портить DataFrame inplace)

        X.apply(lambda x: self.type_assert(x), axis=1)

        # создаем признаки наличия аккорда (this realisation helps also to avoid 'PerformanceWarning: DataFrame is highly fragmented.')
        new_chords_cols = pd.DataFrame(np.zeros((X.shape[0], len(self.chords_base))),
                                       columns=list(self.chords_base),
                                       index=X.index).astype(int)
        X = pd.concat([X, new_chords_cols], axis=1)

        # расставляем значения признаков наличия аккордов
        for col in self.chords_columns:
            X[col] = X['chords'].apply(lambda x: 1 if col in x else 0)

        # print(__class__, 'transform OK')
        return X


class NewCalcFeaturesCreator(BaseEstimator, TransformerMixin):
    """
    New calculated features creating
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # print(__class__, 'fit OK')
        return self

    def transform(self, X):
        # print(__class__, 'transform START')

        # кол_во аккордов всего
        X['chords_num'] = X['chords'].apply(lambda x: len(x))

        # кол_во минорных аккордов
        X['minors'] = X['chords'].apply(lambda x: sum('m' in i for i in x))
        # есть ли минорные
        X['is_minors'] = X['minors'] > 0
        # % минорных аккордов
        X['minors_ratio'] = X['minors'] / X['chords_num']

        # кол_во септаккордов
        X['sept'] = X['chords'].apply(lambda x: sum('7' in i for i in x))
        # есть ли септаккорды
        X['is_sept'] = X['sept'] > 0
        # % септаккордов
        X['sept_ratio'] = X['sept'] / X['chords_num']

        # кол_во аккордов со смещенным басом
        X['other_bass'] = X['chords'].apply(lambda x: sum('/' in i for i in x))
        # есть ли аккорды со смещенным басом
        X['is_other_bass'] = X['other_bass'] > 0
        # % аккордов со смещенным басом
        X['other_bass_ratio'] = X['other_bass'] / X['chords_num']

        # не работает идея
        # X = X[X.columns]    # otherwise PerformanceWarning: DataFrame is highly fragmented.

        assert X.isna().sum().sum() == 0, f"ERROR: {X.isna().sum().sum()} NA values!"

        # print(__class__, 'transform OK')
        return X


class FeatureDeletor(BaseEstimator, TransformerMixin):
    """
    Transformer to delete features
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        # print(__class__, 'fit OK')
        return self
    
    def transform(self, X):
        # print(__class__, 'transform START')
        X.drop(columns=self.key, inplace=True)
        # print(__class__, 'transform OK')
        return X


class TEST_PIPELINE(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # print(__class__, 'TEST fit OK')
        return self
    
    def transform(self, X):
        # print(__class__, 'TEST transform OK')
        return X

## pipelines for feature engineering

num_feats = ['chords_num', 'minors', 'is_minors', 'minors_ratio', 'sept', 
             'is_sept', 'sept_ratio', 'other_bass', 'is_other_bass', 'other_bass_ratio',
             ]

TEST_pipeline_ = Pipeline([
    ('test', TEST_PIPELINE()),
])


chords_feats_creator = Pipeline([
    # ('test', TEST_pipeline_),
    ('chords_feats_creator', ChordsFeatureCreator()),
    ('feats_deletor', FeatureDeletor(['date', 'url', 'text'])),
])

calc_feats_creator = Pipeline([
    ('calc_feats_creator', NewCalcFeaturesCreator()),
    ('feats_deletor', FeatureDeletor(['chords', 'view', 'star'])),
])

scaler = Pipeline([
    ('selector', ColumnSelector(num_feats)),
    ('scaler', StandardScaler()),
])

# FeatureUnion
# feats = FeatureUnion([('description', description),
# )]

feats_prepare = Pipeline([
    # ('no_chords_songs_deletor', no_chords_songs_deletor),
    ('chords_feats_creator', chords_feats_creator),
    ('calc_feats_creator', calc_feats_creator),
    ('scaler', scaler),
])

# FeatureUnion
feats = FeatureUnion([('feats_prepare', feats_prepare),
                      ])

## pipeline with model

rf_params = {'n_estimators': 100,
          'random_state': 42}
# RandomForestRegressor(n_estimators=100, *, criterion='squared_error', 
#     max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#     max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
#     bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
#     warm_start=False, ccp_alpha=0.0, max_samples=None)

gb_params = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42}
# GradientBoostingRegressor(*, loss='squared_error', learning_rate=0.1, n_estimators=100, 
#     subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
#     min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, 
#     random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, 
#     warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)


pipeline = Pipeline([
    ('features', feats),
    # ('test', TEST_pipeline_),
    # ('regressor', RandomForestRegressor(**rf_params)),
    ('regressor', GradientBoostingRegressor(**gb_params)),
])

# Посмотрим, как выглядит наш pipeline
# pipeline.steps

## pipeline FIT

### Prepare DataFrame
db_filename = './db/ddt.csv'

raw_df = pd.read_csv(db_filename,
                     encoding='utf-8',
                     verbose=False)

# затычка
raw_df = raw_df.rename(columns={'Unnamed: 0': 'song_name'})
raw_df.set_index('song_name', inplace=True)

# создаем копию для изменений (чтобы не трансформировать исходник)
df_p = raw_df.copy()
# df_p.shape

# преобразуем столбец 'chords' в тип list
# этот класс используем вне пайплайна
df_p = ChordFeatureToListMaker().transform(df_p)
# df_p.shape

# (этот класс используем вне пайплайна, т.к. удалять нужно и в X и в y)
# возможно есть более правильные пути, но я не придумал...
no_chords_songs_deletor = NoChordsSongsDeletor()

# удалим песни без аккордов 
df_p = no_chords_songs_deletor.fit_transform(df_p)
# df_p.shape

# выделяем таргет (сначала рассчитаем его)
target_col = 'song_rating'
df_p[target_col] = df_p['star'] / df_p['view'] * 1000

X_p = df_p.drop(columns=[target_col])
y_p = df_p[target_col]

# X_p.shape, y_p.shape

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, 
                                                            shuffle=True, 
                                                            test_size=0.20, 
                                                            random_state=42)

# X_train_p.shape, X_test_p.shape, y_train_p.shape, y_test_p.shape

##### save to csv

# save test
X_test_p.to_csv("./db/X_test_p.csv", index=None)
y_test_p.to_csv("./db/y_test_p.csv", index=None)

# save train
X_train_p.to_csv("./db/X_train_p.csv", index=None)
y_train_p.to_csv("./db/y_train_p.csv", index=None)

# y_train_p.isna().sum()

### **fit ALL**

# %%time
pipeline.fit(X_train_p, y_train_p)

#### save pipeline to dill
with open("./app/models/rf_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)

### **predict ALL**

y_preds = pipeline.predict(X_test_p)
# pipeline.score()
print(y_preds[:10])

### predict for 1

# предсказание для 1 песни
# idx = 0
# sample_song = pd.DataFrame(X_train_p.iloc[idx:idx+1])  # DF из одной песни
# print(sample_song)
# y_preds_1 = pipeline.predict(sample_song)[0]
# print(y_preds_1)

### pipeline score

# score_train = pipeline.score(X_train_p, y_train_p)
# score_test = pipeline.score(X_test_p, y_test_p)
# print(score_train, score_test)
