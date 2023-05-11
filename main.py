from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import geopy.distance

# Define the app
app = FastAPI()
templates = Jinja2Templates(directory="src")

# Define the input data model
class InputData(BaseModel):
    pickup_datetime: str
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

# Load the trained model from a pkl file
with open('static/base_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get('/')
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Define the API endpoint for the prediction
@app.post('/api')
def predict_fare_amount(request: Request, pickup_datetime: str = Form(...), pickup_longitude: float = Form(...), pickup_latitude: float = Form(...), dropoff_longitude: float = Form(...), dropoff_latitude: float = Form(...), passenger_count: int = Form(...)):
    # Create a dataframe containing the input data

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

    # Perform the prediction using the loaded model
    y_pred = model.predict(data)

    # Return the prediction as a JSON response
    return {'fare_amount': round(float(y_pred[0]), 2)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

