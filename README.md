# discover_feature_relationships

Attempt to discover 1D relationships between all columns in a DataFrame using scikit-learn (RandomForests) and standard correlation tests (Pearson, Spearman and Kendall via Pandas). 

The goal is to see if we can better understand the data in a DataFrame by learning which features (1 column at a time) predict each other column. This code attempts to learn a predictive relationship between the Cartesian product (all pairs) of all columns.

Rather than just learning which column(s) predict a target column, we might want to know what other relationships exist (e.g. during Exploratory Data Analysis) and whether some predictive features are driven by other less-predictive features (to help us find new & better features or data sources). We might also sense-check out data by checking that certain relationships exist.

By default it assumes every target column is a regression challenge. You can provide a list of columns to treat as classification challenges. For regression we cap negative scores at 0 (r^2 can be arbitrarily negative, we cap at 0 to make this a little easier to interpret). Text-encoded columns are automatically LabelEncoded (this is a sensible default but may not reveal information in your case, you might need to provide your own smarter encoding).

## Titanic example

* Embarked (classification) is predicted well by Fare, also by Age
* Pclass (regression) is predicted by Fare but Fare (regression) is poorly predicted by Pclass
* Sex (classification) is predicted well by Survived
* Survived (classification) is predicted well by Sex, Fare, Pclass, SibSpParch
  * Predicting this feature at circa 0.62 is equivalent to "no information" as 0.62 is the mean of Survived
* SibSpParch is predicted by both SibSp and Parch (SibSpParch is the sum of both - it is an engineered additional feature) - it is also predicted by Fare
* SibSp and Parch are also predicted by Fare (but less well so than by SibSpParch)

![alt text](example_titanic_output.png)

This is generated using:
```
df = pd.read_csv("titanic_train.csv")
...

import discover
df_results = discover.discover(cols, classifier_overrides, df)

df_results.pivot(index='target', columns='feature', values='score').fillna(1) \
.style.background_gradient(cmap="viridis", low=0.3, high=0.0, axis=1) \
.set_precision(2)
```

## Boston example

[Boston Notebook](./example_boston_discover_feature_relationships.ipynb)

* NOX predicts AGE and DIS (but not the other way around)
* target predicts LSTAT, LSTAT weakly predicts target, LSTAT weakly predicts RM
* DIS predicts AGE, AGE weakly predicts DIS
* INDUS predicts CRIM and somewhat AGE, B
* target weakly predicts RM, RM weakly predicts target

# Requirements

* scikit-learn (0.19+)
* pandas
* jupyter notebook

# Tests

* Run `discover.py` for a simple test that the code is working 
* Run `pytest` to run `test_discover.py` for a single unit test (use `pytest -s` to see `print` outputs)

# TODO

* how many nans dropped?
* use auc not accuracy for classifier
* for Titanic graph some of these relationships
* add `all_1s_` additional column to simulate "no information"
* use CV rather than train_test_split

# Note to Ian

Environment: `. ~/anaconda3/bin/activate discover_feature_relationships`

