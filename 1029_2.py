from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math
import pandas as pd


speed = 50
max_dist = 3000  #maximum_distance
time = max_dist/speed

with open('test1_popn.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

df = pd.read_csv('Route_Distances.csv', delimiter=r',\s+', index_col=0)
print(df.to_dict())


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

            locations = [(row[3]), row[4]]


    print (locations1)

    """
    locations = \
        [(-95.436960, 29.779630),  # RSS, 7777 Washington Ave
                (-95.52307, 30.020329), ( -95.359396, 29.931701),  # row 0
                (-95.805101, 29.796431), (-95.219793, 29.598091),
                (-95.533034, 29.932055), (-95.113964, 29.659826),
                (-95.113964, 29.659826), (-95.245786, 29.595033),
                (-95.914379, 30.075356), (-95.28119, 30.12724),
                (-95.264803, 30.1144), (-95.54983, 30.72724),
                (-95.419336, 30.018464), (-95.172707, 29.981498),
                (-95.686076, 29.911038), (-95.063668, 29.900089)]
    """
    num_locations = len(locations1)
    dist_matrix = {}
    capacities = [ range(3500, 3700)]

    data["locations"] = [(l[0] * 1, l[1] * 1) for l in locations1]
    data["num_locations"] = len(locations1) #len(data["locations"])
    data["num_vehicles"] = 15
    data["depot"] = 0
    data["demands"] = popn
    data["vehicle_capacities"] = capacities

    # Implementing Constraints




###################################
# Distance callback and dimension #
####################################

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


  return (lat_d + lon_d)


def create_distance_callback(data):
  #Creates callback to return distance between points
  _distances = {}

  for from_node in range(data["num_locations"]):
    _distances[from_node] = {}
    for to_node in range(data["num_locations"]):
      if from_node == to_node:
        _distances[from_node][to_node] = 0
      else:
        _distances[from_node][to_node] = (
            route_distance(data["locations"][from_node],
                               data["locations"][to_node]))

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
                data["locations"][node_index],
                data["locations"][next_node_index])
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
        data["num_locations"],
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
