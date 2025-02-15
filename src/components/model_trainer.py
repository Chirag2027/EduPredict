import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# Make a Config file first

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# Class responsible for training my model
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # Initiate Model training
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting Training & Test input Data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Create a dictionary of models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=0),
                "AdaBoost": AdaBoostRegressor()
            }

            # HyperParameter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Classifier":{},
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report, trained_models = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            if isinstance(model_report, tuple):
                model_report, trained_models = model_report

            # Get Best Model name from dict
            best_model_name = max(model_report, key=lambda x : model_report[x]["test_score"])

            # Best score
            best_model_score = model_report[best_model_name]["test_score"]

            best_model = trained_models[best_model_name]



            if best_model_score < 0.6:
                raise CustomException("No Best Model Found!!")

            logging.info("Best MODEL found on both training  & testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # predicted output for the test data
            predicted = best_model.predict(X_test)
            r2_score_obt = r2_score(y_test, predicted)
            return r2_score_obt

        except Exception as e:
            raise CustomException(e, sys)



