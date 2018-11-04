from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math

with open('test1_popn.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

def create_data():
    # Locations
    num_vehicles = 5
    depot = 0
    locations = \
        [(-95.436960, 29.779630),  # RSS, 7777 Washington Ave
         (-95.52307, 30.020329), (-95.359396, 29.931701),  # row 0
         (-95.805101, 29.796431), (-95.219793, 29.598091),
         (-95.533034, 29.932055), (-95.113964, 29.659826),
         (-95.113964, 29.659826), (-95.245786, 29.595033),
         (-95.914379, 30.075356), (-95.28119, 30.12724),
         (-95.264803, 30.1144), (-95.54983, 30.72724),
         (-95.419336, 30.018464), (-95.172707, 29.981498),
         (-95.686076, 29.911038), (-95.063668, 29.900089)]
    num_locations = len(locations)
    dist_matrix = {}

    for from_node in range(num_locations):
        dist_matrix[from_node] = {}

        for to_node in range(num_locations):
            dist_matrix[from_node][to_node] = (
                route_distance(
                    locations[from_node],
                    locations[to_node]))
    """"    
    with open('test1_popn.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.
    
    class AutoVivification(dict):
    #Implementation of perl's autovivification feature.

    def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    data = AutoVivification()
    filename = 'Route_Distances2.csv'
    with open(filename, 'r+') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip the header
        for row in reader:
            data[row[1]][row[4]] = row[5]

    """
    
    return [num_vehicles, depot, locations, dist_matrix]








###################################
# Distance callback and dimension #
####################################

def route_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""

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

def CreateDistanceCallback(dist_matrix):

  def dist_callback(from_node, to_node):
    return dist_matrix[from_node][to_node]

  return dist_callback

"""
def create_demand_callback(data):
    #Creates callback to get demands at each location.
    def demand_callback(from_node, to_node):
        return data["demands"][from_node]
    return demand_callback

def add_capacity_constraints(routing, data, demand_callback):
    #Adds capacity constraint"
    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0, # null capacity slack
        data["vehicle_capacities"], # vehicle maximum capacities
        True, # start cumul to zero
        capacity)
"""
def add_distance_dimension(routing, distance_callback):
  """Add Global Span constraint"""
  distance = 'Distance'
  maximum_distance = 3000  # Maximum distance per vehicle.
  routing.AddDimension(
      distance_callback,
      0,  # null slack
      maximum_distance,
      True,  # start cumul to zero
      distance)
  distance_dimension = routing.GetDimensionOrDie(distance)
  # Try to minimize the max distance among vehicles.
  distance_dimension.SetGlobalSpanCostCoefficient(100)
###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Print routes on console."""
  total_distance = 0
  for vehicle_id in range(data["num_vehicles"]):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    distance = 0
    while not routing.IsEnd(index):
      plan_output += ' {} ->'.format(routing.IndexToNode(index))
      previous_index = index
      index = assignment.Value(routing.NextVar(index))
      distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
    plan_output += ' {}\n'.format(routing.IndexToNode(index))
    plan_output += 'Distance of route: {}m\n'.format(distance)
    print(plan_output)
    total_distance += distance
  print('Total distance of all routes: {}m'.format(total_distance))
########
# Main #
########
def main():
  """Entry point of the program"""
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
  add_distance_dimension(routing, distance_callback)
  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  print_solution(data, routing, assignment)
if __name__ == '__main__':
  main()
"""
def add_distance_dimension(routing, dist_callback):
  #Add Global Span constraint
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



####################
# Get Routes Array #
####################
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

########
# Main #
########

def main():
  #Entry point of the program
  # Instantiate the data problem.
  [num_vehicles, depot, locations, dist_matrix] = create_data()
  num_locations = len(locations)
  # Create Routing Model
  routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
  # Define weight of each edge
  dist_callback = CreateDistanceCallback(dist_matrix)
  routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
  add_distance_dimension(routing, dist_callback)
  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  routes = get_routes_array(assignment, num_vehicles, routing)
  print("Routes array:")
  print(routes)

if __name__ == '__main__':
  main()

"""
"""
###########
# Printer #
###########
def print_solution(data, routing, assignment):
    #Print routes on console.
    total_dist = 0
    for vehicle_id in xrange(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += manhattan_distance(
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
########
# Main #
########
def main():
  #Entry point of the program
  #Instantiate the data problem.
  data = create_data_model()
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
  print_solution(data, routing, assignment)
if __name__ == '__main__':
 #main()
"""