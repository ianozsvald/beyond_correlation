# discover_feature_relationships

Attempt to discover 1D relationships between all columns in a DataFrame using scikit-learn (RandomForests). 

The goal is to see if we can better understand the data in a DataFrame by learning which features (1 column at a time) predict each other column. This code attempts to learn a predictive relationship between the Cartesian product (all pairs of columns) of all columns.

By default it assumes every target column is a regression challenge. You can provide a list of columns to treat as classification challenges.

![alt text](example_titanic_output)

# TODO

  * how many nans dropped?
  * use auc not accuracy for classifier

# Note to Ian

Environment: `. ~/anaconda3/bin/activate discover_feature_relationships`

