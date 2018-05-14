from sklearn.model_selection import GridSearchCV

# Parameter Tuning - Lasso
parameters = {
    'alpha': [0.1, 1, 10],
    'normalize': [True, False]
}
score = 'rmse'
clf = GridSearchCV(Lasso(), parameters, cv = 5, scoring = 'neg_mean_squared_error')
clf.fit(train_red, y_train_red)
params_lasso = clf.best_estimator_

# Parameter Tuning - XGB
parameters = {
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
    'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.01, 0.015, 0.025, 0.05, 0.1, 1.0]
}
score = 'rmse'
clf = GridSearchCV(xgb.XGBRegressor(), parameters, cv = 5, scoring = 'neg_mean_squared_error')
clf.fit(train_red, y_train_red)
params_xgb = clf.best_estimator_