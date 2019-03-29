from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd



i = 0
transposed_row = []
popn = []
podid =[]



df = pd.read_csv('long_lat.csv')

list1 = []

for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = []
    p = list(a)
    k = []
    k.append(row['longitude'])
    k.append(row['latitude'])

    for x in k:
        p.append(x)

    list1.append((p))

loc1 = list1

def haversine(lon1, lat1, lon2, lat2):


    R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c

print(loc1)
ResultArray = array(loc1)

N = ResultArray.shape[0]
distance_matrix = zeros((N, N))
for i in range(N):
    for j in range(N):
        lati, loni = ResultArray[i]
        latj, lonj = ResultArray[j]
        distance_matrix[i, j] = haversine(loni, lati, lonj, latj)
        distance_matrix[j, i] = distance_matrix[i, j]


print (distance_matrix)
