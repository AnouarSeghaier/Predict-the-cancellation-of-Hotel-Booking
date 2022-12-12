import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap
import plotly.express as px
import sort_dataframeby_monthorweek as sd
import warnings
from warnings import filterwarnings
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
filterwarnings("ignore")

df = pd.read_csv('venv/data/hotel_booking.csv')
df=df.drop(['name', 'email','phone-number', 'credit_card'], axis=1)
print(df.head())
print(df.shape)
print(df.isna().sum())

def data_clean (df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())

data_clean(df)
print(df.columns)
list=['adults','children','babies']

for i in list:
    print('{} has a unique values {}'.format(i,df[i].unique()))

pd.set_option('display.max_columns',32)

filter=(df['adults']==0) & (df['children']==0) & (df['babies']==0)
print(df[filter])
data=df[~filter]
print(data.head())

country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','No of guests']
print(country_wise_data)

map=folium.Map()
map=px.choropleth(country_wise_data,
              locations=country_wise_data['country'],
              color=country_wise_data['No of guests'],
              hover_name=country_wise_data['country'],
              title='Home country of guests')
map.show()

data2=data[data['is_canceled']==0]
plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.title('prise of room type per night and per person')
plt.xlabel('Room type')
plt.ylabel('price(euro)')
plt.legend()
plt.show()

data_resort = data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel')&(data['is_canceled']==0)]

resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()

final = resort_hotel.merge(city_hotel, on='arrival_date_month')
final.columns=['month','price_for_resort','price_for_city_hotel']
print(final)

def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)

final = sort_data(final,'month')

graph=px.line(final,x='month',y=['price_for_resort','price_for_city_hotel'],title='room priceper night over the months')
graph.show()

rush_resort = data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns =['month', 'no of guests']
print(rush_resort)

rush_city = data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns =['month', 'no of guests']
print(rush_city)

final_rush =rush_resort.merge(rush_city,on='month')
print(final_rush)

final_rush.columns=['month','no of guests in resort','no of guests in city hotel']
final_rush=sort_data(final_rush,'month')
print(final_rush)

rush_grf =px.line(final_rush,x='month',y=['no of guests in resort','no of guests in city hotel'],title='total no of guests per months')
rush_grf.show()

co_relation=data.corr()['is_canceled']
print(co_relation.abs().sort_values(ascending=False))

print(data.groupby('is_canceled')['reservation_status'].value_counts())

list_not=['days_in_waiting_list','arrival_date_year']

num_features=[col for col in data.columns if data[col].dtype !='O' and col not in list_not]
print(num_features)

cat_not=['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']
cat_features=[col for col in data.columns if data[col].dtype =='O' and col not in cat_not]
print(cat_features)

data_cat=data[cat_features]
print(data_cat.dtypes)

data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])

data_cat['year']=data_cat['reservation_status_date'].dt.year
data_cat['month']=data_cat['reservation_status_date'].dt.month
data_cat['day']=data_cat['reservation_status_date'].dt.day

data_cat.drop('reservation_status_date',axis=1,inplace=True)
data_cat['cancellation']=data['is_canceled']

cols =data_cat.columns[0:8]
for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict)

dataframe=pd.concat([data_cat,data[num_features]],axis=1)
dataframe.drop('cancellation',axis=1,inplace=True)
print(dataframe.shape)

sns.displot(dataframe['lead_time'])
plt.show()

def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])

handle_outlier('lead_time')
sns.displot(dataframe['lead_time'])
plt.show()

sns.displot(dataframe['adr'])
plt.show()
handle_outlier('adr')
sns.displot(dataframe['adr'])
plt.show()

dataframe.dropna(inplace=True)

y=dataframe['is_canceled']
x=dataframe.drop('is_canceled',axis=1)

features_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))

features_sel_model.fit(x,y)
print(features_sel_model.get_support())

cols=x.columns
selected_feat=cols[features_sel_model.get_support()]
print('total_featureas {}'.format(x.shape[1]))
print('selected_featureas {}'.format(len(selected_feat)))
