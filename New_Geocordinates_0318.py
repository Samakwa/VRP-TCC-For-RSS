from __future__ import print_function
from math import *
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv


# Problem Data Definition #


speed = 50
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

Dist_matrix = []



i = 0
transposed_row = []
popn = []
podid =[]

import pandas as pd

df = pd.read_csv('long_lat.csv')

list1 = []

for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = ()
    p = list(a)
    k = []
    k.append(row['longitude'])
    k.append(row['latitude'])

    for x in k:
        p.append(x)

    list1.append(tuple(p))


#print(list1)

with open('long_lat.csv') as csvDataFile:
#with open('source_population.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader)
    for row in csvReader:
        podid.append(int(row[0]))

loc1 = list1
def create_data():
  """Stores the data for the problem"""
  # Locations
  num_vehicles = 4
  depot = 0
  locations = loc1
  """"
                [(4, 4), # depot
                 (2, 0), (8, 0), # row 0
                 (0, 1), (1, 1),
                 (5, 2), (7, 2),
                 (3, 3), (6, 3),
                 (5, 5), (8, 5),
                 (1, 6), (2, 6),
                 (3, 7), (6, 7),
                 (0, 8), (7, 8)]
                 
                 
"""
  num_locations = len(locations)
  dist_matrix = {}

  for from_node in range(num_locations):
    dist_matrix[from_node] = {}

    for to_node in range(num_locations):
      dist_matrix[from_node][to_node] = (
        haversine_distance(
          locations[from_node],
          locations[to_node]))

  return [num_vehicles, depot, locations, dist_matrix]

###################################
# Distance callback and dimension #
###################################

def haversine_distance(pointA, pointB):

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = pointA[0]
    lon1 = pointA[1]

    lat2 = pointB[0]
    lon2 = pointB[1]

    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2]) 

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3958.8    #6371  Radius of earth in kilometers. Use 3956 for miles
    return c * r
"""
def haversine_distance(position_1, position_2):
  #Computes the haversine distance between two points
  return (abs(position_1[0] - position_2[0]) +
          abs(position_1[1] - position_2[1]))
"""
def CreateDistanceCallback(dist_matrix):

  def dist_callback(from_node, to_node):
    return dist_matrix[from_node][to_node]

  return dist_callback


def add_distance_dimension(routing, dist_callback):
  """Add Global Span constraint"""
  distance = "Distance"
  maximum_distance = 3000
  routing.AddDimension(
    dist_callback,
    0, # null slack
    maximum_distance, # maximum distance per vehicle
    True, # start cumul to zero
    distance)
  distance_dimension = routing.GetDimensionOrDie(distance)
  # Try to minimize the max distance among vehicles.
  distance_dimension.SetGlobalSpanCostCoefficient(100)

#Print

def print_routes(num_vehicles, locations, routing, assignment):
  """Prints assignment on console"""
  total_dist = 0

  for vehicle_id in range(num_vehicles):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
    route_dist = 0

    while not routing.IsEnd(index):
      node = routing.IndexToNode(index)
      next_node = routing.IndexToNode(
        assignment.Value(routing.NextVar(index)))
      route_dist += haversine_distance(
        locations[node],
        locations[next_node])
      plan_output += ' {node} -> '.format(
        node=node)
      index = assignment.Value(routing.NextVar(index))

    node = routing.IndexToNode(index)
    total_dist += route_dist
    plan_output += ' {node}\n'.format(
      node=node)
    plan_output += 'Distance of route {0}: {dist}\n'.format(
      vehicle_id,
      dist=route_dist)
    print(plan_output)
  print('Total distance of all routes: {dist}'.format(dist=total_dist))

########
# Main #
########

def main():
  """Entry point of the program"""
  # Create data.
  [num_vehicles, depot, locations, dist_matrix] = create_data()
  num_locations = len(locations)
  # Create Routing Model
  start_locations = [0]
  end_locations = podid
  routing = pywrapcp.RoutingModel(num_locations, num_vehicles, start_locations,end_locations)

  dist_callback = CreateDistanceCallback(dist_matrix)
  add_distance_dimension(routing, dist_callback)
  routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  print_routes(num_vehicles, locations, routing, assignment)

if __name__ == '__main__':
  main()