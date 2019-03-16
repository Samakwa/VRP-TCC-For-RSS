from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import sys
import threading
sys.setrecursionlimit(100000)
threading.stack_size(200000000)

#thread = threading.Thread #(target=your_code)
#thread.start(routing_enums_pb2)

speed = 50
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

Dist_matrix = []



i = 0
transposed_row = []
popn = []

with open('first25distances.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader)
    for row in csvReader:
        if row[0] != str(i):
            c = transposed_row.copy()
            Dist_matrix.append(c)
            i = i + 1
            del transposed_row[:]
            transposed_row.append(float(row[5]))

        else:

            transposed_row.append(float(row[5]))

        # del transposed_row[:]
        # transposed_row.append(row[5])
        # i=i+1

with open('first25demand.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader)
    for row in csvReader:
        popn.append(int(row[2]))


c = transposed_row.copy()
Dist_matrix.append(c)
del transposed_row[:]

Dist_matrix.pop(0)
for location in Dist_matrix:
    print (location)
#print (Dist_matrix)
print ("Population", popn)


def create_data_model():
  #Stores the data for the problem
  data = {}

  _distances = Dist_matrix
          #[(4, 4), locations2]

  demands = popn

  #capacities = [3600, 3600, 1000, 3600, 3600, 3600, 3600, 3600, 3600, 3600] # 3600, 3600, 3600, 3600, 3600]
  capacities = [211200, 150,000, 211200, 211200,211200] #, 22,  22, 15, 22,22,22,22,22,22,22]

  data["distances"] = _distances
  data["num_locations"] = len(_distances)
  data["num_vehicles"] = 5
  data["depot"] = 0
  data["demands"] = demands
  data["vehicle_capacities"] = capacities
  return data

def create_distance_callback(data):

  distances = data["distances"]

  def distance_callback(from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return distances[from_node][to_node]
  return distance_callback

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

def create_demand_callback(data):
    """Creates callback to get demands at each location."""
    def demand_callback(from_node, to_node):
        return data["demands"][from_node]
    return demand_callback

def add_capacity_constraints(routing, data, demand_callback):
    #Adds capacity constraint
    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0, # null capacity slack
        data["vehicle_capacities"], # vehicle maximum capacities
        True, # start cumul to zero
        capacity)

def print_solution(data, routing, assignment):
  """Print routes on console."""
  total_distance = 0
  for vehicle_id in range(data["num_vehicles"]):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    route_dist = 0
    while not routing.IsEnd(index):
      node_index = routing.IndexToNode(index)
      next_node_index = routing.IndexToNode(
        assignment.Value(routing.NextVar(index)))
      route_dist += routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)
      plan_output += ' {0} ->'.format(node_index)
      index = assignment.Value(routing.NextVar(index))
    plan_output += ' {}\n'.format(routing.IndexToNode(index))
    plan_output += 'Distance of route: {}m\n'.format(route_dist)
    print(plan_output)
    total_distance += route_dist
  print('Total distance of all routes: {}m'.format(total_distance))

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
  # Add Capacity constraint
  demand_callback = create_demand_callback(data)
  #add_capacity_constraints(routing, data, demand_callback)
  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member
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