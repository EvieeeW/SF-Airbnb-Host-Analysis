import pandas as pd
import numpy as np
import matplotlib                  
import matplotlib.pyplot as plt
import seaborn as sns             
import geopandas as gpd            
plt.style.use('fivethirtyeight')
%matplotlib inline

import folium
import folium.plugins

import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

# dataset overview
df_listing = pd.read_csv('/Users/wang/Desktop/AirbnbData/listings.csv')

# deal with the missing data
df_listing.isnull().sum()
# visualization
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df_listing.isnull(), yticklabels=False, cbar=False)
plt.show()

# delete the columns with too many missing values
df_listing = df_listing.drop(['neighbourhood_group', 'license'], axis = 1)

# fill other missing values
df_listing = df_listing.fillna(0)

# correlation
corr = df_listing.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

