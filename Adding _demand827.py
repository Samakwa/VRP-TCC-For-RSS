from __future__ import print_function
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import networkx as nx
import csv
import webbrowser


def distance(x1, y1, x2, y2):
    dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

    return dist
class Vehicle():
    """Stores the property of a vehicle"""
    def __init__(self):
        """Initializes the vehicle properties"""
        self._capacity = 17

    @property
    def capacity(self):
        """Gets vehicle capacity"""
        return self._capacity

class CityBlock():
    """City block definition"""
    @property
    def width(self):
        """Gets Block size West to East"""
        return 228/2

    @property
    def height(self):
        """Gets Block size North to South"""
        return 80

class DataProblem():
    """Stores the data for the problem"""
    def __init__(self):
        """Initializes the data for the problem"""
        self._vehicle = Vehicle()
        self._num_vehicles = 5

        # Locations in block unit
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

        with open('test1_popn.csv', 'r+') as in_file:
            OurPOD = csv.reader(in_file)
            has_header = csv.Sniffer().has_header(in_file.readline())
            in_file.seek(0)  # Rewind.
            if has_header:
                next(OurPOD)  # Skip header row.
            """
            for row in OurPOD:
                x = row[2]
                y = row[3]

                # print(x)
                #x = float(x)
                #y = float(y)
                locations = [(x,y)]
          """
                # locations in meters using the city block dimension
        city_block = CityBlock()
        self._locations = [(
        loc[0]*city_block.width,
        loc[1]*city_block.height) for loc in locations]

        self._depot = 0


        #for row in OurPOD:
            #self._demands = int (row[4])


        self._demands =  pd.read_csv('https://gist.githubusercontent.com/Samakwa/b553b392c7104960202a42bc262c7960/raw/240968919a8d95ed3504d88a28d80cab50bdf04b/Popn1.csv')
        """ [0, #RSS
             1, 1,
             2, 4,
             2, 4,
             8, 8,
             1, 2,
             1, 2,
             4, 4,
             8, 8]
        """
    @property
    def vehicle(self):
        """Gets a vehicle"""
        return self._vehicle

    @property
    def num_vehicles(self):
        """Gets number of vehicles"""
        return self._num_vehicles

    @property
    def locations(self):
        """Gets locations"""
        return self._locations

    @property
    def num_locations(self):
        """Gets number of locations"""
        return len(self.locations)

    @property
    def depot(self):
        """Gets depot location index"""
        return self._depot

    @property
    def demands(self):
        """Gets demands at each location"""
        return self._demands



#def distance(x1, y1, x2, y2):
 #   dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)


def distance_n(position_1, position_2):
    #Computes the distance between two points
    return (abs(position_1[0] - position_2[0]) +
            abs(position_1[1] - position_2[1]))
class CreateDistanceEvaluator(object): # pylint: disable=too-few-public-methods
    """Creates callback to return distance between points."""
    def __init__(self, data):
        """Initializes the distance matrix."""
        self._distances = {}

        # precompute distance between location to have distance callback in O(1)
        for from_node in xrange(data.num_locations):
            self._distances[from_node] = {}
            for to_node in xrange(data.num_locations):
                if from_node == to_node:
                    self._distances[from_node][to_node] = 0
                else:
                    self._distances[from_node][to_node] = (
                        distance_n(
                            data.locations[from_node],
                            data.locations[to_node]))

    def distance_evaluator(self, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return self._distances[from_node][to_node]

class CreateDemandEvaluator(object): # pylint: disable=too-few-public-methods
    #Creates callback to get demands at each location.
    def __init__(self, data):
        """Initializes the demand array."""
        self._demands = data.demands

    def demand_evaluator(self, from_node, to_node):
        """Returns the demand of the current node"""
        del to_node
        return self._demands[from_node]

def add_capacity_constraints(routing, data, demand_evaluator):
    """Adds capacity constraint"""
    capacity = "Capacity"
    routing.AddDimension(
        demand_evaluator,
        0, # null capacity slack
        data.vehicle.capacity, # vehicle maximum capacity
        True, # start cumul to zero
        capacity)

#Printer
class ConsolePrinter():
    """Print solution to console"""
    def __init__(self, data, routing, assignment):
        """Initializes the printer"""
        self._data = data
        self._routing = routing
        self._assignment = assignment

    @property
    def data(self):
        """Gets problem data"""
        return self._data

    @property
    def routing(self):
        """Gets routing model"""
        return self._routing

    @property
    def assignment(self):
        """Gets routing model"""
        return self._assignment

    def print(self):
        """Prints assignment on console"""
        # Inspect solution.
        total_dist = 0
        for vehicle_id in xrange(self.data.num_vehicles):
            index = self.routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
            route_dist = 0
            route_load = 0
            while not self.routing.IsEnd(index):
                node_index = self.routing.IndexToNode(index)
                next_node_index = self.routing.IndexToNode(
                    self.assignment.Value(self.routing.NextVar(index)))
                route_dist += distance_n(
                    self.data.locations[node_index],
                    self.data.locations[next_node_index])
                route_load += self.data.demands[node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                index = self.assignment.Value(self.routing.NextVar(index))

            node_index = self.routing.IndexToNode(index)
            total_dist += route_dist
            plan_output += ' {0} Load({1})\n'.format(node_index, route_load)
            plan_output += 'Distance of the route: {0}m\n'.format(route_dist)
            plan_output += 'Load of the route: {0}\n'.format(route_load)
            print(plan_output)
        print('Total Distance of all routes: {0}m'.format(total_dist))

#Main

def main():
    #Creating the data problem
    data = DataProblem()

    # Create Routing Model
    routing = pywrapcp.RoutingModel(data.num_locations, data.num_vehicles, data.depot)
    # Define weight of each edge
    distance_evaluator = CreateDistanceEvaluator(data).distance_evaluator
    routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator)
    # Add Capacity constraint
    demand_evaluator = CreateDemandEvaluator(data).demand_evaluator
    add_capacity_constraints(routing, data, demand_evaluator)

    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    printer = ConsolePrinter(data, routing, assignment)
    printer.print()

"""
def openmap():
  with open('newroutes.csv', 'w') as out_file:
        new_list = csv.writer(out_file)

    webbrowser.open("https://planner.myrouteonline.com/route-planner")
    webbrowser.open("https://www.google.es/maps/dir/'-95.436960,29.779630'/'-95.063668,29.900089")
"""
if __name__ == '__main__':
  main()

