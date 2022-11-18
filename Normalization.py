import pandas as pd
from sklearn.preprocessing import StandardScaler

def norm(x):
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return x