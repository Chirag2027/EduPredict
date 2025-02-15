import os, sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# dill is the library used to create the pkl file

def save_object(file_path, obj):
    # Save the object to a file
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# def evaluate_model(X_train, y_train, X_test, y_test, models, param):
#     try:
#         report = {}  

#         for model_name, model in models.items():  # Unpack key-value pairs correctly
#             # Train the model
#             model.fit(X_train, y_train)

#             # Make predictions
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             # Calculate r2 scores
#             train_model_score = r2_score(y_train, y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)

#             # Store the test score in the report dictionary
#             report[model_name] = test_model_score  

#         return report  # Ensure the function returns the report

#     except Exception as e:
#         raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}  # Dictionary to store trained models

        for model_name, model in models.items():
            # Hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(model, param.get(model_name, {}), cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best trained model
            best_model = grid_search.best_estimator_

            # Save trained model for later use
            trained_models[model_name] = best_model

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate r2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store model performance in report dictionary
            report[model_name] = {
                "best_params": grid_search.best_params_,
                "train_score": train_model_score,
                "test_score": test_model_score
            }

        return report, trained_models  # Return both performance report and trained models

    except Exception as e:
        raise Exception(f"Error in model evaluation: {str(e)}")

# loading the pickle file for prediction
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)