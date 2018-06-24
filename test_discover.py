import numpy as np
import pandas as pd
from discover import discover

def test1():
    """Exercise a 3 column dataframe to check 2 relationships"""
    X = pd.DataFrame({'a': np.ones(10),
                      'b': np.arange(0, 10),
                      'c': np.arange(0, 20, 2)})
    df_results = discover(X)
    print(df_results)
    assert (df_results.query("feature=='b' and target=='a'")['score'].iloc[0]) == 1, "Expect b to predict a"
    assert (df_results.query("feature=='a' and target=='b'")['score'].iloc[0]) == 0, "Expect a not to predict b" 