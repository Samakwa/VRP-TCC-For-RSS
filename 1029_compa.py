from __future__ import print_function
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv

###########################
# Problem Data Definition #
###########################

num_vehicles = 10



def create_data_model():
    class AutoVivification(dict ):
        """Implementation of perl's autovivification feature."""

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

        """Stores the data for the problem"""

        #data["locations"] = [(l[0] * 114, l[1] * 80) for l in _locations]
        data["num_locations"] = len(data["locations"])
        data["num_vehicles"] = 4
        data["depot"] = 0
        return data

#######################
# Problem Constraints #
#######################

def manhattan_distance(position_1, position_2):
  """Computes the Manhattan distance between two points"""
  return (
      abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1]))

with open('Route_Distances2.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        source = []
        for row in OurPOD:
            source_ID = row[0]
            source_name = row[1]
            Source_Popn = row[2]
            destin_ID = row[3]
            Destin_Name = row[4]
            Distn = row[5]
            source.append(row[2])
            if row[0] == '152':
                print (source)

            dist = float(row[5])


def create_distance_callback(data):
  """Creates callback to return distance between points."""
  _distances = {}

  with open('Route_Distances2.csv', 'r+') as in_file:
      OurPOD = csv.reader(in_file)
      has_header = csv.Sniffer().has_header(in_file.readline())
      in_file.seek(0)  # Rewind.
      if has_header:
          next(OurPOD)  # Skip header row.

      source = []
      for row in OurPOD:
          source_ID = row[0]
          source_name = row[1]
          Source_Popn = row[2]
          destin_ID = row[3]
          Destin_Name = row[4]
          Distn = row[5]
          source.append(row[2])
          if row[0] == '152':
              print(source)

          dist = float(row[5])

          from_node = row[1]
          to_node = row[4]
          for from_node in xrange(data["num_locations"]): _distances[from_node] = {}
          for to_node in xrange(data["num_locations"]):
            if from_node == to_node:
              _distances[from_node][to_node] = 0
            else:
             _distances[from_node][to_node] = (
            manhattan_distance(data["locations"][from_node],
                               data["locations"][to_node]))

  def distance_callback(from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return _distances[from_node][to_node]

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
###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Print routes on console."""
  total_distance = 0
  for vehicle_id in xrange(data["num_vehicles"]):
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
  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  print_solution(data, routing, assignment)
if __name__ == '__main__':
  main()