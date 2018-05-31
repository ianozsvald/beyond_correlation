# discover_feature_relationships

Attempt to discover 1D relationships between all columns in a DataFrame using scikit-learn (RandomForests). 

The goal is to see if we can better understand the data in a DataFrame by learning which features (1 column at a time) predict each other column. This code attempts to learn a predictive relationship between the Cartesian product (all pairs of columns) of all columns.

Rather than just learning which column(s) predict a target column, we might want to know what other relationships exist (e.g. during Exploratory Data Analysis) and whether some predictive features are driven by other less-predictive features (to help us find new & better features or data sources). We might also sense-check out data by checking that certain relationships exist.

By default it assumes every target column is a regression challenge. You can provide a list of columns to treat as classification challenges.

## Titanic example

Here's an example output from 
```
df = pd.read_csv("titanic_train.csv")
...

import discover
df_results = discover.discover(cols, classifier_overrides, df)

df_results.pivot(index='target', columns='feature', values='score').fillna(1) \
.style.background_gradient(cmap="viridis", low=0.3, high=0.0, axis=1) \
.set_precision(2)
```

Notably:

* Embarked (classification) is predicted well by Fare, also by Age
* Pclass (regression) is predicted by Fare
* Fare (regression) is poorly predicted by Pclass
* Sex (classification) is predicted well by Survived
* Survived (classification) is predicted well by Sex, Fare, Pclass, SibSpParch
* SibSp, Parch and SibSpParch (the sum of both) each predict each other

![alt text](example_titanic_output.png)

# Requirements

* scikit-learn (0.19+)
* pandas
* jupyter notebook

# TODO

  * how many nans dropped?
  * use auc not accuracy for classifier

# Note to Ian

Environment: `. ~/anaconda3/bin/activate discover_feature_relationships`

