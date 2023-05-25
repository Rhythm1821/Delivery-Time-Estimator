import os,sys,pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object,Time_diff_bw_order_and_pickup,str_min

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            # print(features)
            data_scaled = preprocessor.transform(features)
            # print(data_scaled)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.info('Error occured while predicting')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,Delivery_person_Age,
                 Delivery_person_Ratings,Delivery_location_latitude
                 ,Delivery_location_longitude,Vehicle_condition,
                 multiple_deliveries,Time_Orderd,Time_Order_picked,
                 Weather_conditions,Road_traffic_density,Type_of_order,
                 Type_of_vehicle,Festival,City):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Time_Orderd = Time_Orderd
        self.Time_Order_picked = Time_Order_picked
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City

    def get_data_as_frame(self):
        try:
            custom_data_input_dict  = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City],
                'Time_Orderd':[self.Time_Orderd],
                'Time_Order_picked':[self.Time_Order_picked],
                # 'Weather_conditions':[self.Weather_conditions],
                # 'Road_traffic_density':[self.Road_traffic_density],
                # 'Type_of_order':[self.Type_of_order],
                # 'Type_of_vehicle':[self.Type_of_vehicle],
                # 'Festival':[self.Festival],
                # 'City':[self.City]
            }
            df = pd.DataFrame(custom_data_input_dict)
            df['Time_Orderd_min'] = df['Time_Orderd'].fillna(df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            df['Time_Order_picked_min'] = df['Time_Order_picked'].fillna(df['Time_Orderd'].mode()[0]).str.split(':').apply(str_min)
            df.drop(columns=['Time_Orderd','Time_Order_picked'],axis=1,inplace=True)
            Time_diff_bw_order_and_pickup(df)
            logging.info('Data Gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)