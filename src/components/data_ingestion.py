import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion starts')

        try:
            df = pd.read_csv(os.path.join('notebooks\data\data.csv'))
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=True)
            logging.info('Train test split')
            train_split,test_split=train_test_split(df,test_size=0.3,random_state=42)

            train_split.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_split.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion completed')

            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    print('Working just fine')