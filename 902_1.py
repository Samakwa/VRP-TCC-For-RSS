import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import networkx as nx
from sklearn import preprocessing

def distance(x1, y1, x2, y2):
    dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

    return dist


class CreateDistanceCallback(object):
  '''Create callback to calculate distances between points.'''

  def __init__(self, G):
      '''Calculate shortest paths using Floyd-Warshall'''
      self.paths = nx.floyd_warshall(G)

  def Distance(self, from_node, to_node):
      return self.paths[from_node][to_node]

d
class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""

  def __init__(self, locations):
    """Initialize distance array."""
    size = len(locations)
    self.matrix = {}

    for from_node in range(size):
      self.matrix[from_node] = {}
      for to_node in range(size):
        x1 = locations[from_node][0]
        y1 = locations[from_node][1]
        x2 = locations[to_node][0]
        y2 = locations[to_node][1]
        self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)

  def Distance(self, from_node, to_node):
    return int(self.matrix[from_node][to_node])

# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at each location."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

def main():
  # Create the data.
  data = create_data_array()
  locations = data[0]
  demands = data[1]
  num_locations = len(locations)
  depot = 0    # The depot is the start and end point of each route.
  num_vehicles = 1

  # Create routing model.
  if num_locations > 0:
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Callback to the distance function.
    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Put a callback to the demands.
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Add a dimension for demand.
    slack_max = 0
    vehicle_capacity = 1500
    fix_start_cumul_to_zero = True
    demand = "Demand"
    routing.AddDimension(demands_callback, slack_max, vehicle_capacity,
                         fix_start_cumul_to_zero, demand)

    # Solve, displays a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
      # Display solution.
      # Solution cost.
      print("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(vehicle_nbr)
        index_next = assignment.Value(routing.NextVar(index))
        route = ''
        route_dist = 0
        route_demand = 0

        while not routing.IsEnd(index_next):
          node_index = routing.IndexToNode(index)
          node_index_next = routing.IndexToNode(index_next)
          route += str(node_index) + " -> "
          # Add the distance to the next node.
          route_dist += dist_callback(node_index, node_index_next)
          # Add demand.
          route_demand += demands[node_index_next]
          index = index_next
          index_next = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        node_index_next = routing.IndexToNode(index_next)
        route += str(node_index) + " -> " + str(node_index_next)
        route_dist += dist_callback(node_index, node_index_next)
        print("Route for vehicle " + str(vehicle_nbr) + ":\n\n" + route + "\n")
        print("Distance of route " + str(vehicle_nbr) + ": " + str(route_dist))
        print("Demand met by vehicle " + str(vehicle_nbr) + ": " + str(route_demand) + "\n")
    else:
      print('No solution found.')
  else:
    print('Specify an instance greater than 0.')
def create_data_graph():
    edgelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv')
    nodelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/nodelist_sleeping_giant.csv')

    node_dict = dict(zip(nodelist['id'], list(range(nodelist.shape[0]))))

    G = nx.Graph()

    for i, elrow in edgelist.iterrows():
        G.add_edge(node_dict[elrow.node1], node_dict[elrow.node2], weight=elrow.distance)

    locations = [[e.X, e.Y] for e in nodelist.itertuples()]
    demands = [1] + [1] + [1] * 75


if __name__ == '__main__':
  main()
