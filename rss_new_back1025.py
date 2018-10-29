import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def distance(x1, y1, x2, y2):

    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return dist

def create_data_array():
    locations = open('Route_Distances2.csv', 'r+')
    return locations


# Distance callback

class CreateDistanceCallback(object):
    """Create callback to calculate distances between points."""

    def __init__(self):
        """Initialize distance array."""
        locations = create_data_array()
        size = len(locations)
        self.matrix = {}

        for from_node in range(size):
            self.matrix[from_node] = {}
            for to_node in range(size):
                if from_node == to_node:
                    self.matrix[from_node][to_node] = 0
                else:
                    x1 = locations[from_node][0]
                    y1 = locations[from_node][1]
                    x2 = locations[to_node][0]
                    y2 = locations[to_node][1]
                    self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)

    def Distance(self, from_node, to_node):
        return int(self.matrix[from_node][to_node])

def main():

    # Create the data.
    locations = create_data_array()
    tsp_size = len(locations)
    num_routes = 1      # The number of routes, which is 1 in the TSP.
    # Nodes are indexed from 0 to tsp_size - 1. The depot is the starting node of the route.
    depot = 0
    # Create routing model.
    if tsp_size > 0:
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # Set search time limit in milliseconds.
        search_parameters.time_limit_ms = 5000
        # Add Pickup and Delivery constraints
        routing.AddPickupAndDelivery(1,2)
        routing.AddPickupAndDelivery(2,3)
        # Callback to the distance function. The callback takes two
        # arguments (the from and to node indices) and returns the distance between them.
        dist_between_locations = CreateDistanceCallback()
        dist_callback = dist_between_locations.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        # Solve, returns a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:

            # Solution cost.
            print ("Total distance: " + str(assignment.ObjectiveValue()) + "\n")

            # Inspect solution.
            # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1.
            route_number = 0
            node = routing.Start(route_number)
            start_node = node
            route = ''

            while not routing.IsEnd(node):
                route += str(node) + ' -> '
                node = assignment.Value(routing.NextVar(node))
            route += '0'
            print ("Route:\n\n" + route)
        else:
            print ('No solution found.')
    else:
        print ('Specify an instance greater than 0.')

if __name__ == '__main__':
    main()