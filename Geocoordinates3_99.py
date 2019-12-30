
from __future__ import print_function
from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

speed = 70
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

distance_matrix1 = []

popn = [] 
podid =[]


df = pd.read_csv('LGA_coordinates.csv')

list1 = []

for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = []
    p = list(a)
    k = []
    #demand1 =[]
    k.append(row['long'])
    k.append(row['lat'])
    popn.append(row['population'])
    #k.append(row['id'])
    #k.append(row['address'])
    #k.append(row['city'])
    #k.append(str(row['zip']))

    for x in k:
        p.append(x)

    list1.append((p))


loc1 = list1


def haversine(lon1, lat1, lon2, lat2):


    R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c

print(loc1)
ResultArray = array(loc1)

N = ResultArray.shape[0]
distance_matrix1 = zeros((N, N))
for i in range(N):
    for j in range(N):
        lati, loni, *_ = ResultArray[i]
        latj, lonj, *_ = ResultArray[j]
        distance_matrix1[i, j] = haversine(float(loni), float(lati), float(lonj), float(latj))
        distance_matrix1[j, i] = distance_matrix1[i, j]

print ("Distance Matrix:")
print (distance_matrix1)

print ("Popn:", popn)


"""
def create_data_model():
  #Stores the data for the problem
  data = {}
  _distances = distance_matrix1
          #[(4, 4), locations2]

  demands = popn
  capacities = [

      300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
      300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
  ]


  data["distances"] = _distances
  data["num_locations"] = len(_distances)
  data["num_vehicles"] = 25
  data["depot"] = 0
  data["demands"] = demands
  data["vehicle_capacities"] = capacities
  data["time_per_demand_unit"] = 30
  data["vehicle_speed"] = 70
  return data

"""
def create_data_model():
    #Stores the data for the problem.
    
    
    data = {}
    data['distance_matrix'] = distance_matrix1




    data['demands'] = popn
    data['vehicle_capacities'] = [900000, 900000, 900000, 900000, 900000, 900000, 900000, 900000]
    data['num_vehicles'] = 8
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, assignment):
    #Prints assignment on console.
    total_distance = 0
    total_load = 0
    #time_dimension = routing.GetDimensionOrDie('Time')
    #total_time = 0

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        time_d =route_distance/speed
        print ("time: ", time_d)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def create_time_evaluator(data):
    """Creates callback to get total times between locations."""

    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data['demands'][node] * data['time_per_demand_unit']

    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = manhattan_distance(data['locations'][from_node], data[
                'locations'][to_node]) / data['vehicle_speed']
        return travel_time

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in range(len(data['num_locations'])):
        _total_time[from_node] = {}
        for to_node in range(len(data['num_locations'])):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            else:
                _total_time[from_node][to_node] = int(
                    service_time(data, from_node) + travel_time(
                        data, from_node, to_node))

    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes"""
        return _total_time[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return time_evaluator
def main():

    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()