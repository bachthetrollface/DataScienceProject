from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, accuracy_score, precision_score, \
    recall_score, f1_score, median_absolute_error
from pandas import DataFrame, Series

def cross_val_metrics_calculate(model, X:DataFrame, y:Series, splits, metrics=['mse', 'rmse', 'mae', 'mape', 'medae', 'medape']):
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
        if 'accuracy' in metrics:
            result['accuracy'] += accuracy_score(y_test, y_pred)
        if 'precision' in metrics:
            result['precision'] += precision_score(y_test, y_pred, average='macro', zero_division=0)
        if 'recall' in metrics:
            result['recall'] += recall_score(y_test, y_pred, average='macro', zero_division=0)
        if 'f1' in metrics:
            result['f1'] += f1_score(y_test, y_pred, average='macro', zero_division=0)
        if 'medae' in metrics:
            result['medae'] += median_absolute_error(y_test, y_pred)
        if 'medape' in metrics:
            result['medape'] += median_absolute_percentage_error(y_test, y_pred)
    for metric in metrics:
        result[metric] /= n_folds
    return result

def median_absolute_percentage_error(y_true, y_pred):
    result = abs(y_true - y_pred) / y_true
    return result.median()