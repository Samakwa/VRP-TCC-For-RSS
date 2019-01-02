from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math
#import pandas as pd


speed = 50
max_dist = 3000  #maximum_distance
time = max_dist/speed

with open('test1_popn.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

#df = pd.read_csv('Route_Distances.csv', delimiter=r',\s+', index_col=0)
#print(df.to_dict())


def create_data():

    num_vehicles = 15
    depot = 0

    locations1 = []
    popn = []
    data = {}
    with open('test1_popn.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.



        for row in OurPOD:
            long = float (row[3])
            lat = float (row[4])
            locations1.append([long, lat])
            popn.append(row[5])

            #locations = [(row[3]), row[4]]


    print (locations1)
    #for item in locations1:
    #    print (item)

    #num_locations = len(locations1)
    #dist_matrix = {}
    capacities = [ 3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600 ]
    """
    for item in locations1:
        for item_1, item_2 in item:
            lat1, lon1, lat2, lon2 = map(math.radians, [position_1[0], position_1[1], position_2[0], position_2[1]])

    position_1 = lat
    position_2 = long
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [locations1[0][0], locations1[0][1], locations1[1][0], locations1[1][1]] )
    print (locations1[0][1])
    dlat = lat2 - lat1
    a1 = math.sin(dlat / 2) ** 2
    c1 = 2 * math.atan2(math.sqrt(a1), math.sqrt(1 - a1))
    dlon = lon2 - lon1
    a2 = math.sin(dlon / 2) ** 2
    c2 = 2 * math.atan2(math.sqrt(a2), math.sqrt(1 - a2))
    r = 6371
    c1 =c1 *r
    c2 = c2 * r
    print (c1,c2)
    #new=
    data["locations1"] =   [(l[0] * (c2* r), l[1] * (c1*r)) for l in locations1]
    data["num_locations1"] = len("locations1") # len[data[locations1])
    data["num_vehicles"] = 15
    data["depot"] = 0
    data["demands"] = popn
    data["vehicle_capacities"] = capacities

    return data
    # Implementing Constraints




###################################
# Distance callback and dimension #
####################################

"""
def route_distance(position_1, position_2):
    #computes distance between two points

  # convert decimal degrees to radians

  lat1, lon1, lat2, lon2 = map(math.radians, [position_1[0], position_1[1], position_2[0], position_2[1]])

  # haversine formula for delta_lat
  dlat = lat2 - lat1
  a = math.sin(dlat / 2) ** 2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  # r = 6371km
  r = 3959  # Miles
  lat_d = c * r

  # haversine formula for delta_lon
  dlon = lon2 - lon1
  a = math.sin(dlon / 2) ** 2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  r = 6371
  lon_d = c * r

  #print (lat_d + lon_d)
  return (lat_d + lon_d)

"""

def route_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""
  return (
      abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1]))
def create_distance_callback(data):
  #Creates callback to return distance between points
  _distances = {}

  for from_node in range(data["num_locations1"]):
    _distances[from_node] = {}
    for to_node in range(data["num_locations1"]):
      if from_node == to_node:
        _distances[from_node][to_node] = 0
      else:
        _distances[from_node][to_node] = (
            route_distance(data["locations1"][from_node],
                               data["locations1"][to_node]))

  def distance_callback(from_node, to_node):

    return _distances[from_node][to_node]

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
        0, # null capacity slack
        data["vehicle_capacities"], # vehicle maximum capacities
        True, # start cumul to zero
        capacity)

def print_solution(data, routing, assignment):
    """Print routes on console."""
    total_dist = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += route_distance(
                data["locations1"][node_index],
                data["locations1"][next_node_index])
            route_load += data["demands"][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        total_dist += route_dist
        plan_output += ' {0} Load({1})\n'.format(node_index, route_load)
        plan_output += 'Distance of the route: {0}m\n'.format(route_dist)
        plan_output += 'Load of the route: {0}\n'.format(route_load)
        print(plan_output)
    print('Total Distance of all routes: {0}m'.format(total_dist))

def main():
    #Entry point of the program
    # Instantiate the data problem.
    data = create_data()
    # Create Routing Model
    routing = pywrapcp.RoutingModel(
        data["num_locations1"],
        data["num_vehicles"],
        data["depot"])
    # Define weight of each edge
    distance_callback = create_distance_callback(data)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
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
