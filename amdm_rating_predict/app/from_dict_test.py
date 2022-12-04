import pandas as pd

request_json = {'chords': "['C','D','A','C']",
                'view': '5555',
                'star': '55',
                'date': '0',
                'url': '0',
                'text': '0'
                }

data_dict = request_json
# data_dict = {k: [v] for (k, v) in request_json.items()}
df_ = pd.DataFrame.from_dict(data_dict, orient='index').T
df_.iloc[0] = df_.iloc[0].apply(lambda x: eval(x))


print(df_)
el = df_.iloc[0]['chords']
# el = df_.iloc[0]['star']
# el = eval(el)
print(type(el))
# print(type(eval(el)))
