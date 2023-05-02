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
from src.utils import save_object,Time_diff_bw_order_and_pickup,str_min


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation started')
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings','Delivery_location_latitude', 'Delivery_location_longitude',
            'Vehicle_condition', 'multiple_deliveries','Time_diff_bw_order_and_pickup']
            categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
            'Type_of_vehicle', 'Festival', 'City']

            categories = ['Fog','Stormy','Sandstorms','Windy','Cloudy','Sunny'],['Jam', 'High', 'Medium', 'Low'],['Drinks','Snack','Meal','Buffet'],['bicycle','electric_scooter','scooter','motorcycle'],['Yes','No'],['Semi-Urban','Urban','Metropolitian']

            logging.info('Pipeline initiated')

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
            logging.info('Pipeline completed')  
            return preprocessor
            

        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info('Read train and test data')
            logging.info(f'Train DataFrame head \n{train_df.head().to_string}')
            logging.info(f'Test DataFrame head \n{test_df.head().to_string}')

            logging.info('Training data cleaning starts')

            train_df.drop(columns=['Order_Date','ID','Restaurant_latitude','Restaurant_longitude','Delivery_person_ID'],axis=1,inplace=True)

            train_df['Time_Orderd_min'] = train_df['Time_Orderd'].fillna(train_df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            train_df['Time_Order_picked_min'] = train_df['Time_Order_picked'].fillna(train_df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            train_df.drop(columns=['Time_Orderd','Time_Order_picked'],axis=1,inplace=True)

            Time_diff_bw_order_and_pickup(train_df)

            logging.info('Training data cleaning completed')

            logging.info('Testing data cleaning starts')

            test_df.drop(columns=['Order_Date','ID','Restaurant_latitude','Restaurant_longitude','Delivery_person_ID'],axis=1,inplace=True)

            test_df['Time_Orderd_min'] = test_df['Time_Orderd'].fillna(test_df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            test_df['Time_Order_picked_min'] = test_df['Time_Order_picked'].fillna(test_df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            test_df.drop(columns=['Time_Orderd','Time_Order_picked'],axis=1,inplace=True)

            Time_diff_bw_order_and_pickup(test_df)

            logging.info('Testing data cleaning completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_object = self.get_data_transformation_object()

            input_feature_train_df = train_df.drop(columns=['Time_taken (min)'],axis=1)
            target_feature_train_df = train_df['Time_taken (min)']

            input_feature_test_df = test_df.drop(columns=['Time_taken (min)'],axis=1)
            target_feature_test_df = test_df['Time_taken (min)']

            #Transforming using preprocessor
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.fit_transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing data')

            train_arr = np.c_[input_feature_train_arr,np.array(input_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception occured in initiate data transformation')
            raise CustomException(e,sys)
        
