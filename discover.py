from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def labelencode_if_object(df_ml):
    for col in df_ml.columns:
        if df_ml[col].dtype == 'O':
            le = LabelEncoder()
            replacement_series = le.fit_transform(df_ml[col])
            #print("dropping", col)
            df_ml = df_ml.drop(columns=[col])
            df_ml[col] = replacement_series
    return df_ml

def discover(cols, classifier_overrides, df):
    estimator_mapping = {}
    for col in cols:
        if col in classifier_overrides:
            est = RandomForestClassifier()
        else:
            est = RandomForestRegressor()
        estimator_mapping[col] = est
    
    ds = []
    for idx_Y, target in enumerate(cols):
        est = estimator_mapping[target]
        for idx_X, feature in enumerate(cols):
            if idx_X == idx_Y:
                continue

            df_ml = df[[feature, target]]
            rows_before_drop_na = df_ml.shape[0]
            df_ml = df_ml.dropna()
            rows_after_drop_na = df_ml.shape[0]
            #if rows_after_drop_na < rows_before_drop_na:
            #    print(feature, target)
            #    print(f"Dropped {rows_before_drop_na - rows_after_drop_na} rows")

            df_ml = labelencode_if_object(df_ml)

            df_X = df_ml[[feature]]
            df_y = df_ml[target]

            assert df_X.isnull().sum().sum() == 0
            assert df_y.isnull().sum() == 0

            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=0)

            #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            est.fit(X_train, y_train)
            score = est.score(X_test, y_test)
            d = {'feature': feature, 'target': target, 'score': score}
            ds.append(d)

    df_results = pd.DataFrame(ds)
    return df_results