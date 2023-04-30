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
            logging.info('Data transformation started')
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings','Delivery_location_latitude', 'Delivery_location_longitude',
            'Vehicle_condition', 'multiple_deliveries','Time_diff_bw_order_and_pickup']
            categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
            'Type_of_vehicle', 'Festival', 'City']

            categories = ['Fog','Stormy','Sandstorms','Windy','Cloudy','Sunny'],['Jam', 'High', 'Medium', 'Low'],['Drinks','Snack','Meal','Buffet'],['bicycle','electric_scooter','scooter','motorcycle'],['Yes','No'],['Semi-Urban','Urban','Metropolitian']

            #Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            #Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=list(categories))),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical pipeline', num_pipeline,numerical_columns),
                ('categorical pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor
            logging.info('Pipeline completed')

        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            pass
            # train_df= pd.read_csv('')
        except Exception as e:
            logging.info('Exception occured in initiate data transformation')
            raise CustomException(e,sys)
        
if __name__=='__main__':
    print('Working just fine')        
