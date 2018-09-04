from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from six.moves import xrange
import math



def routes_info(assignment, num_vehicles, routing):
  # Get the routes for an assigment and return as a list of lists.
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


#assignment = routing.SolveWithParameters(search_parameters)
#routes = routes_info(assignment, num_routes, routing)

def distance(x1, y1, x2, y2):
    dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

#def distance(x1, y1, x2, y2):
    #dist = abs(x1 - x2) + abs(y1 - y2)

    return dist


class CreateDemandCallback(object):
  #Create call back to get demands at each location.

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

"""
class CreateDistanceCallback(object):
  '''Create callback to calculate distances between points.'''

  def __init__(self, G):
      '''Calculate shortest paths using Floyd-Warshall'''
      self.paths = nx.floyd_warshall(G)

  def Distance(self, from_node, to_node):
      return self.paths[from_node][to_node]
"""


class CreateDistanceCallback(object):
  """Create callback to calculate distances and travel times between points."""

  def __init__(self, locations):
    """Initialize distance array."""

    num_locations = len(locations)
    self.matrix = {}

    for from_node in xrange(num_locations):
      self.matrix[from_node] = {}
      for to_node in xrange(num_locations):
        x1 = locations[from_node][0]
        y1 = locations[from_node][1]
        x2 = locations[to_node][0]
        y2 = locations[to_node][1]
        self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)


  def Distance(self, from_node, to_node):
    return int(self.matrix[from_node][to_node])

"""
vehicle_load_time = 180
vehicle_unload_time = 180
solver = routing.solver()
intervals = []
for i in range(num_vehicles):
  # Add time windows at start of routes
  intervals.append(solver.FixedDurationIntervalVar(routing.CumulVar(routing.Start(i), time),
                                                     vehicle_load_time,
                                                     "depot_interval"))
  # Add time windows at end of routes.
  intervals.append(solver.FixedDurationIntervalVar(routing.CumulVar(routing.End(i), time),
                                                     vehicle_unload_time,
                                                     "depot_interval"))
"""
def create_data():
  """Stores the data for the problem"""
  # Locations
  num_vehicles = 4
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
        manhattan_distance(
          locations[from_node],
          locations[to_node]))

  return [num_vehicles, depot, locations, dist_matrix]

###################################
# Distance callback and dimension #
###################################


def manhattan_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""
  return (abs(position_1[0] - position_2[0]) +
          abs(position_1[1] - position_2[1]))

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
      route_dist += manhattan_distance(
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
  start_locations = [8, 3, 15, 7]
  end_locations = [11, 10, 1, 6]
  routing = pywrapcp.RoutingModel(num_locations, num_vehicles, start_locations, end_locations)

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
