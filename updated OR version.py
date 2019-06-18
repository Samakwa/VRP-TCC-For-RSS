from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import math
import threading
import webbrowser
from matplotlib import pyplot as plt
from collections import namedtuple



distance_matrix = []

popn = []
podid =[]


df = pd.read_csv('PodData_MM.csv')

list1 = []

for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = []
    p = list(a)
    k = []
    #demand1 =[]
    k.append(row['longitude'])
    k.append(row['latitude'])
    popn.append(row['population'])
    k.append(row['id'])
    k.append(row['address'])
    k.append(row['city'])
    k.append(str(row['zip']))

    for x in k:
        p.append(x)

    list1.append((p))


loc1 = list1


def haversine( lat1, long1, lat2,long2):


    R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

    degrees_to_radians = math.pi / 180.0
    phi1 = lat1 * degrees_to_radians
    phi2 = lat2 * degrees_to_radians
    lambda1 = long1 * degrees_to_radians
    lambda2 = long2 * degrees_to_radians
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = haversine2(dphi) + math.cos(phi1) * math.cos(phi2) * haversine2(dlambda)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def haversine2(angle):
    h = math.sin(angle / 2) ** 2
    return h

print(loc1)
ResultArray = array(loc1)

N = ResultArray.shape[0]
distance_matrix = zeros((N, N))
for i in range(N):
    for j in range(N):
        longi, lati, *_ = ResultArray[i]
        longj, latj, *_ = ResultArray[j]
        distance_matrix[i, j] = haversine(float(longi), float(lati), float(longj), float(latj))
        distance_matrix[j, i] = distance_matrix[i, j]


print (distance_matrix)
print ("Popn:", popn)




def create_data_model():
  """Stores the data for the problem."""
  data = {}
  data['time_matrix'] = [
      distance_matrix
  ]
  data['time_windows'] = [
      (0, 5),  # depot
      (7, 12),  # 1
      (10, 15),  # 2
      (16, 18),  # 3
      (10, 13),  # 4
      (0, 5),  # 5
      (5, 10),  # 6
      (0, 4),  # 7
      (5, 10),  # 8
      (0, 3),  # 9
      (10, 16),  # 10
      (10, 15),  # 11
      (0, 5),  # 12
      (5, 10),  # 13
      (7, 8),  # 14
      (10, 15),  # 15
      (11, 15),  # 16
  ]
  data['num_vehicles'] = 4
  data['depot'] = 0
  return data

def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(
            manager.IndexToNode(index), assignment.Min(time_var),
            assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        print(plan_output)
        total_time += assignment.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))

def main():

    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        30,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        print_solution(data, manager, routing, assignment)

if __name__ == '__main__':
  main()