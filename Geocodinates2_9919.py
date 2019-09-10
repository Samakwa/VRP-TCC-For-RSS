from __future__ import print_function
import pandas as pd
import numpy as np
import googlemaps
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

gmaps = googlemaps.Client(key='Key')

"""
def calculate_geocodes():
    df = pd.read_csv("LGA_coordinates.csv")

    df['lat'] = pd.Series(np.repeat(0, df.size), dtype=float)
    df['long'] = pd.Series(np.repeat(0, df.size), dtype=float)

    result = np.zeros([df.size, 2])

    #for index, row in df.iterrows():
        # print(row['Address'])
        #geocode_result = gmaps.geocode(row['Address'])[0]
        #lat = (geocode_result['geometry']['location']['lat'])
        #lng = (geocode_result['geometry']['location']['lng'])
        #result[index] = lat, lng
        #df.lat[index] = lat
        #df.long[index] = lng

    print("First step", df)
    coords = df.as_matrix(columns=['lat', 'long'])

    return coords, df
"""

df = pd.read_csv("LGA_coordinates.csv")
coords = df.as_matrix(columns=['lat', 'long'])
coordinates = coords
def calculate_distance_matrix(coordinates, gmaps):
    distance_matrix = np.zeros(
        (np.size(coordinates, 0), np.size(coordinates, 0)))  # create an empty matrix for distance between all locations

    for index in range(0, np.size(coordinates, 0)):
        src = coordinates[index]

        for ind in range(0, np.size(coordinates, 0)):
            dst = coordinates[ind]
            distance_matrix[index, ind] = distance(src[0], src[1], dst[0], dst[1])

    return distance_matrix


def distance(lat1, long1, lat2, long2):
    # Note: The formula used in this function is not exact, as it assumes
    # the Earth is a perfect sphere.

    # Mean radius of Earth in miles
    radius_earth = 3959

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0
    phi1 = lat1 * degrees_to_radians
    phi2 = lat2 * degrees_to_radians
    lambda1 = long1 * degrees_to_radians
    lambda2 = long2 * degrees_to_radians
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = haversine(dphi) + math.cos(phi1) * math.cos(phi2) * haversine(dlambda)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius_earth * c
    return d


def haversine(angle):
    h = math.sin(angle / 2) ** 2
    return h


def create_data_model(distance_matrix, number_of_vehicles, depot):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    print(distance_matrix)

    data['num_vehicles'] = number_of_vehicles
    data['depot'] = depot
    return data


def print_solution(data, manager, routing, solution, address_dataframe):
    """Prints solution on console."""
    max_route_distance = 0

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ---> '.format(address_dataframe.iloc[manager.IndexToNode(index), 0])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))


def main():
    #address_dataframe = calculate_geocodes()
    distance_matrix = calculate_distance_matrix(coordinates, gmaps)
    data = create_data_model(distance_matrix, 5, 0)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])

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

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        80,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 120
    search_parameters.log_search = False

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution) #, address_dataframe)


if __name__ == '__main__':
    main()