import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting data with train and test array')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'SVM':SVR()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            print(model_report)
            print('\n======================================================================')
            logging.info(f'Model Report {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f'Best model: {best_model}, R2Score: {best_model_score}')
            print('\n======================================================================')
            print(model_report)
            # for i in model_report.values():
            #     print(i)
            logging.info(f'Best mode: {best_model}, R2Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured in model trainer')
            raise CustomException(e,sys)