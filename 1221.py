from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math
import pandas as pd
from scipy.spatial import distance_matrix


speed = 50
max_dist = 3000  #maximum_distance
time = max_dist/speed

#pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
#df = pd.from_csv('filename.csv')

dataset = dict()
reader = csv.DictReader(open('Route_Distances2.csv'))

result = {}
for row in reader:
    #key = row.pop('Date')
    if key in result:
        # implement your duplicate row handling here
        pass
    result[key] = row
print (result)
"""
with open('test1_popn.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

df = pd.read_csv('Route_Distances.csv', delimiter=r',\s+', index_col=0)
print(df.to_dict())

"""