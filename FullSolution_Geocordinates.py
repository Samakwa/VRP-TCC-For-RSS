from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import sys
from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd

speed = 50
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

distance_matrix = []

popn = []
podid =[]


df = pd.read_csv('LGA_coordinates.csv')

list1 = []

for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = []
    p = list(a)
    k = []
    #demand1 =[]
    k.append(row['long'])
    k.append(row['lat'])
    popn.append(row['population'])
    #k.append(row['id'])
    #k.append(row['address'])
    #k.append(row['city'])
    #k.append(str(row['zip']))

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
        lati, loni, *_ = ResultArray[i]
        latj, lonj, *_ = ResultArray[j]
        distance_matrix[i, j] = haversine(float(loni), float(lati), float(lonj), float(latj))
        distance_matrix[j, i] = distance_matrix[i, j]

print ("Distance Matrix:")
print (distance_matrix)
t = open("Ord.csv", "w")
"""
for line in distance_matrix:
    res = line.split(None,1)
    ts = str(res)
    t.write(line+'\n')
t.close()
"""
print ("Popn:", popn)




def create_data_model():
  #Stores the data for the problem
  data = {}

  _distances = distance_matrix
          #[(4, 4), locations2]

  demands = popn

  #capacities = [3600, 3600, 1000, 3600, 3600, 3600, 3600, 3600, 3600, 3600] # 3600, 3600, 3600, 3600, 3600]
  capacities = [

      900000, 900000, 900000, 900000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
      300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,



  ]


  data["distances"] = _distances
  data["num_locations"] = len(_distances)
  data["num_vehicles"] = 25
  data["depot"] = 0
  data["demands"] = demands
  data["vehicle_capacities"] = capacities
  data["time_per_demand_unit"] = 30
  data["vehicle_speed"] = 50
  return data


#######################
# Problem Constraints #
#######################
def create_distance_callback(data):
    """Creates callback to return distance between points."""
    distances = data["distances"]

    def distance_callback(from_node, to_node):

        return distances[from_node][to_node]

    return distance_callback


def create_demand_callback(data):
    """Creates callback to get demands at each location."""

    def demand_callback(from_node, to_node):
        return data["demands"][from_node]

    return demand_callback


def add_capacity_constraints(routing, data, demand_callback):
    """Adds capacity constraint"""
    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        capacity)

def add_distance_dimension(routing, distance_callback):
  """Add Global Span constraint"""
  distance = 'Distance'
  maximum_distance = 800  # Maximum distance per vehicle.
  routing.AddDimension(
      distance_callback,
      0,  # null slack
      maximum_distance,
      True,  # start cumul to zero
      distance)
  #distance_dimension = routing.GetDimensionOrDie(distance)
  # Try to minimize the max distance among vehicles.
  #distance_dimension.SetGlobalSpanCostCoefficient(100)

def service_time(data, node):
    """Gets the service time for the specified location."""
    return data["demands"][node] * data["time_per_demand_unit"]

def travel_time(data, from_node, to_node):
    """Gets the travel times between two locations."""

    travel_time =   data["distances"][from_node][to_node] / data["vehicle_speed"]
    return travel_time

def create_time_callback(data):
  """Creates callback to get total times between locations."""
  def service_time(node):
    """Gets the service time for the specified location."""
    return data["demands"][node] * data["time_per_demand_unit"]

  def travel_time(from_node, to_node):
    """Gets the travel times between two locations."""
    travel_time = data["distances"][from_node][to_node] / data["vehicle_speed"]
    return travel_time

  def time_callback(from_node, to_node):
    """Returns the total time between the two nodes"""
    serv_time = service_time(from_node)
    trav_time = travel_time(from_node, to_node)
    return serv_time + trav_time

  return time_callback


###########
# Printer #
###########
def print_solution(data, routing, assignment):
    """Print routes on console."""
    total_dist = 0
    for vehicle_id in range(data["num_vehicles"]):
        url = 'https://google.com/maps/dir'

        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)

            for item in loc1:
               if item[2] == node_index:
                   url += '/' + str(item[3]) + str(item[4]) +  str(item[5])

            route_load = data["demands"][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            index = assignment.Value(routing.NextVar(index))



        node_index = routing.IndexToNode(index)
        total_dist += route_dist
        time = (route_dist *10)/speed + service_time(data,node_index)
        plan_output += ' {0} Load({1})\n'.format(node_index, route_load)
        plan_output += 'Distance of the route: {0}m\n'.format(route_dist)
        plan_output += 'Time of the route: {0}h\n'.format(time)
        plan_output += 'Load of the route: {0}\n'.format(route_load)
        print(plan_output)
        print("url is: {}".format(url))
    print('Total Distance of all routes: {0}m'.format(total_dist))



#Main
def main():
    """Entry point of the program"""
    # Instantiate the data problem.
    data = create_data_model()
    # Create Routing Model
    routing = pywrapcp.RoutingModel(
        data["num_locations"],
        data["num_vehicles"],
        data["depot"])
    # Define weight of each edge
    distance_callback = create_distance_callback(data)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
    add_distance_dimension(routing, distance_callback)
    time_callback = create_time_callback(data)

    # Add Capacity constraint
    demand_callback = create_demand_callback(data)
    add_capacity_constraints(routing, data, demand_callback)
    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        print_solution(data, routing, assignment)


if __name__ == '__main__':
    main()

"""
minimum_allowed_capacity = 22 #Put vehicle capacity
capacity = "Capacity"
routing.AddDimension(
    demand_evaluator, # function which return the load at each location (cf. cvrp.py example)
    0, # null capacity slack
    data.vehicle.capacity, # vehicle maximum capacity
    True, # start cumul to zero
    capacity)
capacity_dimension = routing.GetDimensionOrDie(capacity)
for vehicle in range(0,vehicle_number):
    capacity_dimension.CumulVar(routing.End(vehicle)).RemoveInterval(0, minimum_allowed_capacity)
"""