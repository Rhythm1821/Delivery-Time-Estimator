import os,sys,pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.fit_transformation(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.info('Error occured while predicting')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self):
        pass

    def get_data_as_frame(self):
        try:
            pass
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)