from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math
import pandas as pd


speed = 50
max_dist = 3000  #maximum_distance
time = max_dist/speed



def distance(position_1, position_2):
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



# Distance callback

class CreateDistanceCallback(object):
  """Create callback to calculate distances and travel times between points."""

  def __init__(self, locations1):
    """Initialize distance array."""

    num_locations = len(locations1)
    self.matrix = {}

    for from_node in range(num_locations):
      self.matrix[from_node] = {}
      for to_node in range(num_locations):
        x1 = locations1[from_node][0]
        y1 = locations1[from_node][1]
        x2 = locations1[to_node][0]
        y2 = locations1[to_node][1]
        self.matrix[from_node][to_node] = distance(position_1, position_2)


  def Distance(self, from_node, to_node):
    return int(self.matrix[from_node][to_node])



# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at each node."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

# Service time (proportional to demand) + transition time callback.
class CreateServiceTimeCallback(object):
  """Create callback to get time windows at each node."""

  def __init__(self, demands, time_per_demand_unit):
    self.matrix = demands
    self.time_per_demand_unit = time_per_demand_unit

  def ServiceTime(self, from_node, to_node):
    return self.matrix[from_node] * self.time_per_demand_unit

# Create total_time callback (equals service time plus travel time).
class CreateTotalTimeCallback(object):
  def __init__(self, service_time_callback, dist_callback, speed):
    self.service_time_callback = service_time_callback
    self.dist_callback = dist_callback
    self.speed = speed

  def TotalTime(self, from_node, to_node):
    return self.service_time_callback(from_node, to_node) + self.dist_callback(from_node, to_node) / self.speed

def DisplayPlan(routing, assignment):

  # Display dropped orders.
  dropped = ''

  for order in range(1, routing.nodes()):
    if assignment.Value(routing.NextVar(order)) == order:
      if (dropped.empty()):
        dropped += " %d", order
      else: dropped += ", %d", order

  plan_output = 0
  if not dropped.empty():
    plan_output += "Dropped orders:" + dropped + "\n"

  return plan_output

def main():
  # Create the data.
  data = create_data()
  locations = data[0]
  demands = data[1]
  start_times = data[2]
  num_locations = len(locations)
  depot = 0
  num_vehicles = 5
  search_time_limit = 400000

  # Create routing model.
  if num_locations > 0:

    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to tsp_size - 1. By default the start of
    # a route is node 0.
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Callback to the distance function.

    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance

    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Add capacity dimension constraints.
    vehicle_capacity = 100
    null_capacity_slack = 0
    fix_start_cumul_to_zero = True
    capacity = "Capacity"
    routing.AddDimension(demands_callback, null_capacity_slack, vehicle_capacity,
                         fix_start_cumul_to_zero, capacity)

    # Adding time dimension constraints.
    time_per_demand_unit = 300
    horizon = 24 * 3600
    time = "Time"
    tw_duration = 5 * 3600
    speed = 10

    service_times = CreateServiceTimeCallback(demands, time_per_demand_unit)
    service_time_callback = service_times.ServiceTime
    total_times = CreateTotalTimeCallback(service_time_callback, dist_callback, speed)
    total_time_callback = total_times.TotalTime

    # Note: In this case fix_start_cumul_to_zero is set to False,
    # because some vehicles start their routes after time 0, due to resource constraints.

    fix_start_cumul_to_zero = False
    # Add a dimension for time and a limit on the total time_horizon
    routing.AddDimension(total_time_callback,  # total time function callback
                         horizon,
                         horizon,
                         fix_start_cumul_to_zero,
                         time)

    time_dimension = routing.GetDimensionOrDie("Time")

    for order in range(1, num_locations):
      start = start_times[order]
      time_dimension.CumulVar(routing.NodeToIndex(order)).SetRange(start, start + tw_duration)
    # Add resource constraints at the depot (start and end location of routes).

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
    # Constrain the number of maximum simultaneous intervals at depot.
    depot_capacity = 2
    depot_usage = [1 for i in range(num_vehicles * 2)]
    solver.AddConstraint(
      solver.Cumulative(intervals, depot_usage, depot_capacity, "depot"))
    # Instantiate route start and end times to produce feasible times.
    for i in range(num_vehicles):
      routing.AddVariableMinimizedByFinalizer(routing.CumulVar(routing.End(i), time))
      routing.AddVariableMinimizedByFinalizer(routing.CumulVar(routing.Start(i), time))

    # Solve, displays a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:

      # Solution distance.
      print ("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")
      # Display solution.

      capacity_dimension = routing.GetDimensionOrDie(capacity)
      time_dimension = routing.GetDimensionOrDie(time)

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(int(vehicle_nbr))
        plan_output = 'Route {0}:'.format(vehicle_nbr)

        while not routing.IsEnd(index):
          node_index = routing.IndexToNode(index)
          load_var = capacity_dimension.CumulVar(index)
          time_var = time_dimension.CumulVar(index)
          plan_output += \
                    " {node_index} Load({load}) Time({tmin}, {tmax}) -> ".format(
                        node_index=node_index,
                        load=assignment.Value(load_var),
                        tmin=str(assignment.Min(time_var)),
                        tmax=str(assignment.Max(time_var)))
          index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)  # Convert index to node
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += \
                  " {node_index} Load({load}) Time({tmin}, {tmax})".format(
                      node_index=node_index,
                      load=assignment.Value(load_var),
                      tmin=str(assignment.Min(time_var)),
                      tmax=str(assignment.Max(time_var)))
        print (plan_output)
        print ( "\n")
    else:
      print ( 'No solution found.')
  else:
    print ('Specify an instance greater than 0.')

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
    for item in locations1:
        print (item)

    #num_locations = len(locations1)
    #dist_matrix = {}
    capacities = [ 3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600 ]
    start_times = [28842, 50891, 10351, 49370, 22553, 53131, 8908,
                   56509, 54032, 10883, 60235, 46644, 35674, 30304,
                   39950, 38297, 36273, 52108, 2333, 48986, 44552,
                   31869, 38027, 5532, 57458, 51521, 11039, 31063,
                   38781, 49169, 32833, 7392]

    data["locations1"] = [(l[0] * 1, l[1] * 1) for l in locations1]
    data["num_locations"] = len(data["locations1"]) #len(locations1)
    data["num_vehicles"] = 15
    data["depot"] = 0
    data["demands"] = popn
    data["vehicle_capacities"] = capacities

    #return data



    data = [locations1, popn, start_times]
    return data

if __name__ == '__main__':
  main()