import pandas as pd
import geopy.distance
import numpy as np

def distance(row):
    return geopy.distance.geodesic(row['pickup_coords'], row['dropoff_coords']).miles

if __name__ == '__main__'
	uber = pd.read_csv('../data/uber.csv')
	uber.drop('Unnamed: 0', axis=1, inplace=True)

	uber['pickup_year']=pd.DatetimeIndex(uber['pickup_datetime']).year
	uber['pickup_month']=pd.DatetimeIndex(uber['pickup_datetime']).month
	uber['pickup_day']=pd.DatetimeIndex(uber['pickup_datetime']).day
	uber['pickup_hour']=pd.DatetimeIndex(uber['pickup_datetime']).hour
	uber['pickup_minute']=pd.DatetimeIndex(uber['pickup_datetime']).minute
	uber['pickup_second']=pd.DatetimeIndex(uber['pickup_datetime']).second

	# cycling encoding for pickup_month, pickup_day, pickup_hour, pickup_minute, pickup_second

	uber['pickup_month_sin'] = np.sin(uber['pickup_month']*(2.*np.pi/12))
	uber['pickup_month_cos'] = np.cos(uber['pickup_month']*(2.*np.pi/12))
	uber['pickup_day_sin'] = np.sin(uber['pickup_day']*(2.*np.pi/31))
	uber['pickup_day_cos'] = np.cos(uber['pickup_day']*(2.*np.pi/31))
	uber['pickup_hour_sin'] = np.sin(uber['pickup_hour']*(2.*np.pi/24))
	uber['pickup_hour_cos'] = np.cos(uber['pickup_hour']*(2.*np.pi/24))
	uber['pickup_minute_sin'] = np.sin(uber['pickup_minute']*(2.*np.pi/60))
	uber['pickup_minute_cos'] = np.cos(uber['pickup_minute']*(2.*np.pi/60))
	uber['pickup_second_sin'] = np.sin(uber['pickup_second']*(2.*np.pi/60))
	uber['pickup_second_cos'] = np.cos(uber['pickup_second']*(2.*np.pi/60))

	uber.drop('pickup_month', axis=1, inplace=True)
	uber.drop('pickup_day', axis=1, inplace=True)
	uber.drop('pickup_hour', axis=1, inplace=True)
	uber.drop('pickup_minute', axis=1, inplace=True)
	uber.drop('pickup_second', axis=1, inplace=True)

	uber.drop('pickup_datetime', axis=1, inplace=True)
	uber.drop('key', axis=1, inplace=True)

	uber.drop(uber[uber['pickup_longitude']>90].index, inplace=True)
	uber.drop(uber[uber['pickup_latitude']>90].index, inplace=True)
	uber.drop(uber[uber['dropoff_longitude']>90].index, inplace=True)
	uber.drop(uber[uber['dropoff_latitude']>90].index, inplace=True)

	uber.drop(uber[uber['pickup_longitude']<-90].index, inplace=True)
	uber.drop(uber[uber['pickup_latitude']<-90].index, inplace=True)
	uber.drop(uber[uber['dropoff_longitude']<-90].index, inplace=True)
	uber.drop(uber[uber['dropoff_latitude']<-90].index, inplace=True)

	uber.drop(uber[uber['passenger_count'] > 5].index, axis=0, inplace = True)
	uber.drop(uber[uber['passenger_count'] == 0].index, axis=0, inplace = True)
	uber.drop(uber[uber['fare_amount'] < 2.5].index, axis=0, inplace = True)

	uber.dropna(inplace=True)

	uber['pickup_coords'] = list(zip(uber.pickup_latitude, uber.pickup_longitude))
	uber['dropoff_coords'] = list(zip(uber.dropoff_latitude, uber.dropoff_longitude))

	uber['distance'] = uber.apply(distance, axis=1)
	uber.drop(uber[uber['distance'] > 130].index, axis=0, inplace = True)
	uber.drop(uber[uber['distance'] == 0].index, axis=0, inplace = True)


	uber.drop('pickup_coords', axis=1, inplace=True)
	uber.drop('dropoff_coords', axis=1, inplace=True)

	uber.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

	uber.to_csv('processed_uber.csv')