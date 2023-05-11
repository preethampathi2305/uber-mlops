# use flask to deploy the model

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import geopy.distance
import numpy as np

app = Flask(__name__, template_folder="src")

# load the model
model = pickle.load(open('static/base_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api',methods=['POST'])
def predict():
    # get the data from the POST request.
    pickup_datetime = request.form['pickup_datetime']
    pickup_longitude = float(request.form['pickup_longitude'])
    pickup_latitude = float(request.form['pickup_latitude'])
    dropoff_longitude = float(request.form['dropoff_longitude'])
    dropoff_latitude = float(request.form['dropoff_latitude'])
    passenger_count = int(request.form['passenger_count'])
    
    # create a dataframe

    data = pd.DataFrame([[pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]], columns=['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])

    data['pickup_year']=pd.DatetimeIndex(data['pickup_datetime']).year
    data['pickup_month']=pd.DatetimeIndex(data['pickup_datetime']).month
    data['pickup_day']=pd.DatetimeIndex(data['pickup_datetime']).day
    data['pickup_hour']=pd.DatetimeIndex(data['pickup_datetime']).hour
    data['pickup_minute']=pd.DatetimeIndex(data['pickup_datetime']).minute
    data['pickup_second']=pd.DatetimeIndex(data['pickup_datetime']).second

    # cycling encoding for pickup_month, pickup_day, pickup_hour, pickup_minute, pickup_second

    data['pickup_month_sin'] = np.sin(data['pickup_month']*(2.*np.pi/12))
    data['pickup_month_cos'] = np.cos(data['pickup_month']*(2.*np.pi/12))
    data['pickup_day_sin'] = np.sin(data['pickup_day']*(2.*np.pi/31))
    data['pickup_day_cos'] = np.cos(data['pickup_day']*(2.*np.pi/31))
    data['pickup_hour_sin'] = np.sin(data['pickup_hour']*(2.*np.pi/24))
    data['pickup_hour_cos'] = np.cos(data['pickup_hour']*(2.*np.pi/24))
    data['pickup_minute_sin'] = np.sin(data['pickup_minute']*(2.*np.pi/60))
    data['pickup_minute_cos'] = np.cos(data['pickup_minute']*(2.*np.pi/60))
    data['pickup_second_sin'] = np.sin(data['pickup_second']*(2.*np.pi/60))
    data['pickup_second_cos'] = np.cos(data['pickup_second']*(2.*np.pi/60))

    data.drop('pickup_month', axis=1, inplace=True)
    data.drop('pickup_day', axis=1, inplace=True)
    data.drop('pickup_hour', axis=1, inplace=True)
    data.drop('pickup_minute', axis=1, inplace=True)
    data.drop('pickup_second', axis=1, inplace=True)

    data.drop('pickup_datetime', axis=1, inplace=True)
    # data.drop('key', axis=1, inplace=True)

    # calculate distance
    data['distance'] = data.apply(lambda x: geopy.distance.distance((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],x['dropoff_longitude'])).km, axis=1)

    # drop pickup and dropoff latitude and longitude
    data.drop('pickup_latitude', axis=1, inplace=True)
    data.drop('pickup_longitude', axis=1, inplace=True)
    data.drop('dropoff_latitude', axis=1, inplace=True)
    data.drop('dropoff_longitude', axis=1, inplace=True)

    # make prediction
    prediction = model.predict(data)

    # take the first value of prediction
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
