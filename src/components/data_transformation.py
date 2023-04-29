import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_ingestion = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            pass
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            pass
        except Exception as e:
            logging.info('Exception occured in initiate data transformation')
            raise CustomException(e,sys)
        
if __name__=='__main__':
    print('Working just fine')        