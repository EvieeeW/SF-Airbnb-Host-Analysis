# price
sns.distplot(df_listing["price"])  # 直方图
plt.show()

# relationship btw price and min_nights
sns.scatterplot(x="price", y="minimum_nights",data=df_listing)
plt.show()

# neighborhood
top_neighbourhoods = df_listing["neighbourhood"].value_counts().head(5)
df_top_neighbourhoods = df_listing[df_listing["neighbourhood"].isin(top_neighbourhoods.index)]
sns.countplot(df_top_neighbourhoods["neighbourhood"])
plt.show()

df_top_neighbourhoods_cheap = df_top_neighbourhoods[df_top_neighbourhoods.price < 250]   # 小于250房子较多
plt.figure()
sns.boxplot(x = 'neighbourhood', y = 'price', data=df_top_neighbourhoods_cheap)
plt.show()

# room type
room_type_counts = df_listing['room_type'].value_counts()
pie_df = pd.DataFrame({'count': room_type_counts.values}, index=room_type_counts.index)
pie_df.plot(kind='pie', y='count', figsize=(6, 8), autopct=lambda x: '{:.1f}%'.format(x) if x >= 1 else '', startangle=90)
plt.title('Distribution of Room Types')
plt.show()

# room type in 5 top regions
plt.figure(figsize=(12,6))
sns.countplot(data = df_top_neighbourhoods, x="room_type", hue="neighbourhood")
plt.title("Room Types in Top 5 Neighbourhoods")
plt.show()
df_filtered = df_listing[df_listing['price'] <= 5000]
plt.figure()
sns.catplot(data=df_filtered, x="room_type", y="price")
plt.show()

# Distribution on map
plt.figure(figsize=(15,10))
sns.scatterplot(df_listing.longitude, df_listing.latitude, hue=df_listing.neighbourhood)
plt.show()

# on the real world map
import folium
from folium.plugins import HeatMap
m = folium.Map(location=[37.7749, -122.4194], zoom_start=11)
HeatMap(df_listing[['latitude', 'longitude']].dropna(),
        radius=15,
        gradient={0.4: 'white', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}).add_to(m)

display(m)
