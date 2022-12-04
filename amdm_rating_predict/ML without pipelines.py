# необработано! ПРОСТО СКОПИРОВАЛ из Колаба


df = raw_df.copy()
df.shape

# type(df['chords']) to list
df['chords'] = df['chords'].apply(lambda x: eval(x))

# from functools import partial

# ch_base = []
# # df['chords'].
# partial(print, df['chords'])

# получаем список используемых аккордов во всех песнях
chords_base = get_chords_base(df['chords'])
print(f'Кол_во уникальных аккордов: {len(chords_base)}')

print(chords_base)

# удалить песни без аккордов
mask = df['chords'].apply(lambda x: x == [] )
df.drop(df[mask].index, inplace=True)
df.shape
# df[df['column name'].map(len) < 2]  # механизм применения к каждому элементу столбца

# добавляем признаки наличия аккорда
df[list(chords_base)] = 0

# ВРЕМЕННО удаляем ненужные столбцы
df.drop(columns=['date', 'url', 'text'], inplace=True)

# new features
df['song_rating'] = df['star'] / df['view'] * 1000

# кол_во аккордов всего
df['chords_num'] = df['chords'].apply(lambda x: len(x))

# кол_во минорных аккордов
df['minors'] = df['chords'].apply(lambda x: sum('m' in i for i in x))
# есть ли минорные
df['is_minors'] = df['minors'] > 0
# % минорных аккордов
df['minors_ratio'] = df['minors'] / df['chords_num']

# кол_во септаккордов
df['sept'] = df['chords'].apply(lambda x: sum('7' in i for i in x))
# есть ли септаккорды
df['is_sept'] = df['sept'] > 0
# % септаккордов
df['sept_ratio'] = df['sept'] / df['chords_num']

# кол_во аккордов со смещенным басом
df['other_bass'] = df['chords'].apply(lambda x: sum('/' in i for i in x))
# есть ли аккорды со смещенным басом
df['is_other_bass'] = df['other_bass'] > 0
# % аккордов со смещенным басом
df['other_bass_ratio'] = df['other_bass'] / df['chords_num']


df.tail(3)

df.columns[3:]

# в df2 будем удалять столбцы
# df2 = df.copy()

# chords_columns list
FIRST_CHORD_COLUMN_No = 3
chords_columns = df.columns[FIRST_CHORD_COLUMN_No: FIRST_CHORD_COLUMN_No + len(chords_base)]
print(len(chords_columns), (len(chords_columns) == len(chords_base)), chords_columns, sep='\n')

# расставляем значения признаков наличия аккордов
for col in chords_columns:
    df[col] = df['chords'].apply(lambda x: 1 if col in x else 0)

df.tail(2)


# выделяем таргет (сначала рассчитаем его)
target_col = 'song_rating'
df[target_col] = df['star'] / df['view'] * 1000

# сделать умное дропанье (учесть, что song_rating = star / view)
X = df.drop(columns=['chords', 
                      'view', 
                      'star',
                    #   'song_rating',
                      target_col]
             )
y = df[target_col]
X

y

### StandardScaler

# ТОЛЬКО для вещественных признаков !
scaler = StandardScaler()

new_calc_columns = df.columns[-11 + 1:]
new_calc_columns


# transform
for col in new_calc_columns:
    X[col] = scaler.fit_transform(X[[col]])

# X.columns[-1]

# X[X.columns[-1]]

### train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    shuffle=True, 
                                                    test_size=0.20, 
                                                    random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

##### save to csv

# save test
X_test.to_csv("./db/X_test.csv", index=None)
y_test.to_csv("./db/y_test.csv", index=None)

# save train
X_train.to_csv("./db/X_train.csv", index=None)
y_train.to_csv("./db/y_train.csv", index=None)

# Моdels fitting (outside pipelines)

### 1 - LinearRegression

# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)

# y_pred = lr_model.predict(X_test)
# y_pred[:5], y_test[:5]

# evaluate_preds(y_test, y_pred)

# model = lr_model
# print(f'R2 (train): {model.score(X_train, y_train)} \n'
#       f'R2  (test): {model.score(X_test, y_test)} \n'
# )

# распределение коэффициентов ЛР
# отсортировать, построить гистограмму
# lr_model.coef_[:5]

### 2 - Ridge

# ridge_model = Ridge(alpha=0.1)
# ridge_model.fit(X_train, y_train)

# ridge_model.coef_.min(), ridge_model.coef_.max()

# распределение коэффициентов (сделать def!!!)
# coefs = pd.DataFrame(list(zip(ridge_model.feature_names_in_, ridge_model.coef_)),
#                      columns=['feature', 'value'])
# coefs = coefs.set_index('feature')
# coefs['abs_value'] = abs(coefs['value'])
# coefs.sort_values(by='abs_value', ascending=False).head()

# plt.hist(coefs, bins=100);

# y_pred = ridge_model.predict(X_test)
# y_pred[:5], y_test[:5]

# evaluate_preds(y_test, y_pred)

# model = ridge_model
# print(f'R2 (train): {model.score(X_train, y_train)} \n'
#       f'R2  (test): {model.score(X_test, y_test)} \n'
# )

### 3 - Lasso

lasso_model = Lasso(alpha=0.0002)
lasso_model.fit(X_train, y_train)

lasso_model.coef_.min(), lasso_model.coef_.max()

# распределение коэффициентов
plt.hist(lasso_model.coef_, bins=100);

    # видно, что лассо занулило бесполезные признаки

# распределение коэффициентов (сделать def!!!)
coefs = pd.DataFrame(list(zip(lasso_model.feature_names_in_, lasso_model.coef_)),
                     columns=['feature', 'value'])
coefs = coefs.set_index('feature')
coefs['abs_value'] = abs(coefs['value'])
coefs.sort_values(by='abs_value', ascending=False).head()

y_pred = lasso_model.predict(X_test)
y_pred[:5], y_test[:5]

evaluate_preds(y_test, y_pred)

model = lasso_model
print(f'R2 (train): {model.score(X_train, y_train)} \n'
      f'R2  (test): {model.score(X_test, y_test)} \n'
)

