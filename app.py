from flask import Flask,render_template,request,url_for
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Delivery_location_latitude = float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude = float(request.form.get('Delivery_location_longitude')),
            Vehicle_condition = float(request.form.get('Vehicle_condition')),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Time_Orderd = request.form.get('Time_Orderd'),
            Time_Order_picked = request.form.get('Time_Order_picked'),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            Festival = request.form.get('Festival'),
            City = request.form.get('City')
        )
        final_new_data = data.get_data_as_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred,2)

        return render_template('result.html',final_result=results)

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)