import numpy as np
import pandas as pd

X = pd.DataFrame(np.ones((3, 5)))
print(X)
print('='*50)

chords_base = ['A', 'B', 'C']

chords_cols = pd.DataFrame(np.zeros((X.shape[0], len(chords_base))), columns=chords_base)
X = pd.concat([X, chords_cols], axis=1)

print(X.shape)
print(X.columns)
print(X)
