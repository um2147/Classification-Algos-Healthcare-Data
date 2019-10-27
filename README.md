# Classification-Algos-Healthcare-Data

This exploratory data analysis portion was an exercise in identifying the variables that leaked target information in a large dataset. Pre-processing using count across classes was chosen to identify the variables and in later stages, recursive feature elimination was implemented. The baseline model used was the logistic regression model, and using the Target Encoder on select categorical columns was also acceptable.

We moved on to testing other models, such as random forest and xgboost. Random undersampling was employed since the dataset was imbalanced. The gradient boosting performed best when all feature variables are taken in account. 

We moved on to doing feature selection with the aim of getting a simpler and explainable model, and selecting the top 12 and 4 features out of the xg boost model using decision trees gave the best small and explainable model.
