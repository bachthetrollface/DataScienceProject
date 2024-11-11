from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from pandas import DataFrame, Series
def cross_val_metrics_calculate(model, X:DataFrame, y:Series, splits, metrics=['mse', 'rmse', 'mae', 'mape']):
    n_folds = 0
    result = {name:0 for name in metrics}
    for train_index, test_index in splits:
        n_folds += 1
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if 'mse' in metrics:
            result['mse'] += mean_squared_error(y_test, y_pred)
        if 'rmse' in metrics:
            result['rmse'] += root_mean_squared_error(y_test, y_pred)
        if 'mae' in metrics:
            result['mae'] += mean_absolute_error(y_test, y_pred)
        if 'mape' in metrics:
            result['mape'] += mean_absolute_percentage_error(y_test, y_pred)
    for metric in metrics:
        result[metric] /= n_folds
    return result