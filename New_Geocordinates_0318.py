from __future__ import print_function
from math import *
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd

# Problem Data Definition #


speed = 50
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

Dist_matrix = []



i = 0
transposed_row = []
popn = []
podid =[]



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

loc1 = list1

#print(list1)

with open('long_lat.csv') as csvDataFile:
#with open('source_population.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader)
    for row in csvReader:
        podid.append(int(row[0]))
        popn.append(row[4])


def haversine_distance(lon1, lat1, lon2, lat2):


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
        distance_matrix[i, j] = haversine_distance(ResultArray[i], ResultArray[j])
        distance_matrix[j, i] = distance_matrix[i, j]


print (distance_matrix)
def create_data():
  """Stores the data for the problem"""
  # Locations
  data = {}
  num_vehicles = 20
  depot = 0
  locations = loc1
  demands = popn

  num_locations = len(locations)
  dist_matrix = {}

  for from_node in range(0,num_locations):
    dist_matrix[from_node] = {}

    for to_node in range(0,num_locations):
      dist_matrix[from_node][to_node] = (
        haversine_distance(
          locations[from_node],
          locations[to_node]))
  """
  data["distances"] =dist_matrix
  data["num_locations"] = len(dist_matrix)
  data["num_vehicles"] = 6
  data["depot"] = 0
  data["demands"] = demands
  #data["vehicle_capacities"] = capacities
  data["time_per_demand_unit"] = 0.05
  return data
  """
  return [ num_vehicles, depot, locations, dist_matrix]

###################################
# Distance callback and dimension #
###################################

"""
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
def create_dist_callback(dist_matrix):

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
  #distance_dimension.SetGlobalSpanCostCoefficient(100)

def create_demand_callback(demands):
    """Creates callback to get demands at each location."""

    def demand_callback(from_node, to_node):
        return demands [from_node]

    return demand_callback

def get_routes_array(assignment, num_vehicles, routing):
  # Get the routes for an assignent and return as a list of lists.
  routes = []
  for route_nbr in range(num_vehicles):
    node = routing.Start(route_nbr)
    route = []

    while not routing.IsEnd(node):
      index = routing.NodeToIndex(node)
      route.append(index)
      node = assignment.Value(routing.NextVar(node))
    routes.append(route)
  return routes

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
  # Create the data.
  [num_vehicles, depot, locations, dist_matrix] = create_data()
  num_locations = len(locations)

  # Create routing model.
  if num_locations > 0:

    # Set initial routes.
    initial_routes = [[8, 16, 14, 13, 12, 11], [3, 4, 9, 10], [15, 1], [7, 5, 2, 6]]
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    dist_callback = create_dist_callback(dist_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    add_distance_dimension(routing, dist_callback)
    initial_assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)

    print("Initial assignment:\n")
    print_routes(num_vehicles, locations, routing, initial_assignment)

    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    final_assignment = routing.SolveFromAssignmentWithParameters(initial_assignment, search_parameters)

    if final_assignment:
      print("\nFinal assignment:\n")
      print_routes(num_vehicles, locations, routing, final_assignment)
    else:
      print('No solution found.')
  else:
    print('Specify an instance greater than 0.')


if __name__ == '__main__':
  main()