import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for key in models.keys():
            model = models[key]
            para = param.get(key, {})  # Use .get() to avoid KeyError, default to empty dict if not found

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)  # predicting train data
            y_test_pred = model.predict(X_test)  # predicting test data
            train_model_score = r2_score(y_train, y_train_pred)  # calculating r2 score for train data
            test_model_score = r2_score(y_test, y_test_pred)  # calculating r2 score for test data
            report[key] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

