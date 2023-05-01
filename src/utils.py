import sys
import pickle
import os
from src.exception import CustomException


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