import os
import numpy as np
import pandas as pd

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

if __name__ == "__main__":
    print("From URL: ", s)

    print(df.tail())
