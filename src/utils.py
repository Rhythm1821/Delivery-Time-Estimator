import sys
import pickle
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score


def str_min(x):
    min=0
    if type(x)==list:
        if len(x)>1:
            min = x[1]
        else:
            min = int(float(x[0])*60)
    return int(min)


#Calculating time difference order and pickup and removing the minutes column
def Time_diff_bw_order_and_pickup(df):
    x = []
    for i in range(len(df)):
        if df['Time_Order_picked_min'][i] < df['Time_Orderd_min'][i]:
            x.append(df['Time_Order_picked_min'][i] + (60-df['Time_Orderd_min'][i]))
        else:
            x.append(df['Time_Order_picked_min'][i] - df['Time_Orderd_min'][i])
    df['Time_diff_bw_order_and_pickup'] = x
    df.drop(columns=['Time_Orderd_min','Time_Order_picked_min'],axis=1,inplace=True)

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            #Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_pred =model.predict(X_test)

            # Get SVM score for train and test data
            test_model_score = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.info('Exception occured while training model')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured while loading object')