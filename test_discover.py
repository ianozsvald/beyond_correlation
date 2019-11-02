import numpy as np
import pandas as pd
from beyond_correlation.discover import discover

def test1():
    """Exercise a 3 column dataframe to check 2 relationships"""
    X = pd.DataFrame({'a': np.ones(10),
                      'b': np.arange(0, 10),
                      'c': np.arange(0, 20, 2)})
    df_results = discover(X)
    print(df_results)
    assert (df_results.query("feature=='b' and target=='a'")['score'].iloc[0]) == 1, "Expect b to predict a"
    assert (df_results.query("feature=='a' and target=='b'")['score'].iloc[0]) <= 0, "Expect a not to predict b"


def test_na_info():
    """Check a 3 column dataframe with missing data. Make sure number of dropped columns is reported correctly"""
    X = pd.DataFrame({'a': np.ones(10),
                      'b': np.arange(0, 10),
                      'c': np.arange(0, 20, 2)})
    X.iloc[0,0] = np.nan
    X.iloc[2,0] = None
    X.iloc[1,1] = pd.NaT
    _, df_info = discover(X, include_na_information=True)
    assert (df_info.query("feature=='a' and target=='b'")['n_dropped_na'].iloc[0] == 3), "Expect (a,b) to drop 3 rows"
    assert (df_info.query("feature=='c' and target=='b'")['n_dropped_na'].iloc[0] == 1), "Expect (c,b) to drop 1 row"
    assert (df_info.query("feature=='a' and target=='c'")['n_dropped_na'].iloc[0] == 2), "Expect (a,c) to drop 1 row"
