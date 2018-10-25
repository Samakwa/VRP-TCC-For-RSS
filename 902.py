from __future__ import print_function
import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from haversine import haversine

from scipy.spatial import distance
import csv
import numpy as np


def travel_time(data, from_node, to_node):
    """Gets the travel times between two locations."""
    if from_node == to_node:
        travel_time = 0
    else:
        travel_time = r_distance(
            data.locations[from_node],
            data.locations[to_node]) / data.vehicle.speed
    return travel_time

def create_data():
  """Stores the data for the problem"""
  # Locations
  num_vehicles = 6
  RSS = 0

  """ with open('test1.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

    for row in OurPOD:
        x = row[2]
        y = row[3]


        # print(x)
        #x = float(x)
        #y = float(y)
        locations = [(x,y)]
  """
  locations = \
                [(-95.436960, 29.779630), # RSS, 7777 Washington Ave
                 (-95.52307, 30.020329), (-95.359396, 29.931701), # row 0
                 (-95.805101, 29.796431), (-95.219793, 29.598091),
                 (-95.533034, 29.932055), (-95.113964, 29.659826),
                 (-95.113964, 29.659826), (-95.245786, 29.595033),
                 (-95.914379, 30.075356), (-95.28119, 30.12724),
                 (-95.264803, 30.1144), (-95.54983, 30.72724),
                 (-95.419336, 30.018464), (-95.172707, 29.981498),
                 (-95.686076, 29.911038), (-95.063668, 29.900089)]
                 
  """
  locations = \
      [(4, 4),  # depot
       (2, 0), (8, 0),  # row 0
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
        r_distance(
          locations[from_node],
          locations[to_node]))

  return [num_vehicles, RSS, locations, dist_matrix]

###################################
# Distance callback and dimension #
###################################

"""
def r_distance(start, end):
  #Computes the Manhattan distance between two points
  
  with open('test1.csv', 'r+') as in_file:
      OurPOD = csv.reader(in_file)
      has_header = csv.Sniffer().has_header(in_file.readline())
      in_file.seek(0)  # Rewind.
      if has_header:
          next(OurPOD)  # Skip header row.

      for row in OurPOD:
          x = row[5]
          y = row[6]

          # print(x)
          start = float(x)
          end = float(y)

          #origin = (39.50, 98.35)
          #paris = (48.8567, 2.3508)
          #dist =haversine(start, end, miles=True)
          #return dist
          return (abs(start[0] - end[0]) +
          abs(start[1] - end[1]))
"""
def r_distance(position_1, position_2):
  #Computes the Manhattan distance between two points
  return (abs(position_1[0] - position_2[0]) +
          abs(position_1[1] - position_2[1]))

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
  distance_dimension.SetGlobalSpanCostCoefficient(100)

################
# Print Routes #
################

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
      route_dist += r_distance(
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
  [num_vehicles, RSS, locations, dist_matrix] = create_data()
  num_locations = len(locations)

  # Create routing model.
  if num_locations > 0:

    # Set initial routes.
    initial_routes = [8, 16, 14, 13, 12, 11, 3, 4, 9, 10, 15, 1, 7, 5, 2, 6]
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, RSS)
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

