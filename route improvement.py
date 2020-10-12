"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import pywrapcp

from numpy import array,zeros
from math import radians, cos, sin, asin, sqrt
import pandas as pd


speed = 50
max_dist = 3000  #maximum_distance
time =  3000/50 #max_dist/speed

distance_matrix = []

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
distance_matrix = zeros((N, N))
for i in range(N):
    for j in range(N):
        lati, loni, *_ = ResultArray[i]
        latj, lonj, *_ = ResultArray[j]
        distance_matrix[i, j] = haversine(float(loni), float(lati), float(lonj), float(latj))
        distance_matrix[j, i] = distance_matrix[i, j]

print ("Distance Matrix:")
print (distance_matrix)
t = open("Ord.csv", "w")
"""
for line in distance_matrix:
    res = line.split(None,1)
    ts = str(res)
    t.write(line+'\n')
t.close()
"""
print ("Popn:", popn)



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix

    data['initial_routes'] = [
        [8, 16, 14, 13, 12, 11],
        [3, 4, 9, 10],
        [15, 1],
        [7, 5, 2, 6],
    ]
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            #index = int(solution.Value(routing.NextVar(index)))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))




def main():
    """Solve the CVRP problem."""
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
    """
    for node in priority_3_nodes:
        index = manager.NodeToIndex(node)
        routing.NextVar(index).RemoveValues(indices of priority 2 and 1 nodes)
    """
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    initial_solution = routing.ReadAssignmentFromRoutes(data['initial_routes'],
                                                        True)
    print('Initial solution:')
    print_solution(data, manager, routing, initial_solution)

    # Set default search parameters.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # Solve the problem.
    solution = routing.SolveFromAssignmentWithParameters(
        initial_solution, search_parameters)

    # Print solution on console.
    if solution:
        print('Solution after search:')
        print_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()