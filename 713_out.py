import numpy as np
from datetime import date
import csv
import pandas as pd
import math
from math import sqrt
import sys

routes = set()

with open('PODslist.csv', 'r+') as in_file:
    reader = csv.reader(in_file, delimiter='\t')
    PODList = list(reader)


def List2Graph(input_list):
    connections = []
    directions = [(-1,-1),(0,-1),(1,-1),(1,0)]
    for i in range(0,len(input_list)):
        for j in range(0,len(input_list[0])):
            for x,y in directions:
                if (i+y >= 0 and i+y < len(input_list) and j+x >= 0 and j+x < len(input_list)):
                    pair = (input_list[i][j])
                    connections.append(pair)
    return connections

def two_phase():
    #with open('Routes_new.csv', 'w+') as outfile:
    #  writer = csv.writer(outfile)
    #  routes = {rows[0]: rows[1] for rows in reader}

    #PODList = []  # contains a list of all locations to be served
    routes = set()
    opt_weight = 1
    reduction_factor= range(0,1)
    maxDistance = int(input("Enter maximum Distance for this cluster:  "))
    #Assume pod1 is the first location
    ClosestPods = []

    for PodID in PODList:
        #POdID = 1
        # get list of pods ordered by distance
        # implement A Star or Dijkstra to get distance
        initial = PODList[0]
        List2Graph(PODList)
        firstpod = PODList[0][1]
        dijsktra(PODList, initial)
        distance = get_distance(firstpod, PodID)
        temproute1 =[]
        temproute2 = []
        if distance < maxDistance:
            temproute1.append(PodID)
        else:
            temproute2.append(PodID)



        totaldistance =1
        #add cumulative distance
        for start, end in listOfClosestPods:
            if start == end:
                break
            else:

                dijsktra(route, start)
                start= end

            totaldistance+= totaldistance


        prunnedlist =[]
        for PodID in PODList:
            if POdID not in rt1 or PODID not in rt2:
                prunnedlist.append(POdID)




def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distance[(min_node, edge)]
            except:
                continue
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path

def get_distance(start, target):
        """
        Return distance between self and any target object
        """
        x_squared = pow((start.x - target.x), 2)
        y_squared = pow((start.y - target.y), 2)

        return sqrt(x_squared + y_squared)

def prune_route():
    #inputs
    routes = input ("enter or read")
    weight= input ("optimization weight")
    max_duration =input (" Maximum tour duration:")
    total_cap = input ("Maximum vehicle capacity")
    pruned_pods= set()
    pruned_routes = []
    new_routes = set()
    for route in range (0,routes.size -1):
        # put all into a temporary route
        current_cap = input("Current Capacity:")
        current_cap+= current_cap
        if current_cap> total_cap:
            pruned_routes.append(route)
    for route in pruned_routes:
        #create a path through all pruned routes

        if duration < max_duration and pruned_cap< total_cap:
            new_routes.add(route)
        maxDistance = 0.0
        routeStartPod = None
        routeEndPod = None
        for POD.startPod in podsCutFromRoute:
            for PodDistanceNode.endPod in startPod.listOfClosestPods:
                if [podsCutFromRoute, endPod.c_podId] in podList:
                    if endPod.c_distanceToPod >= maxDistance:
                        routeStartPod = startPod
                        routeEndPod = c_allPods.get(endPod.c_podId)
                        maxDistance = endPod.c_distanceToPod


"""
def partition_route():
    #inputs
    locationids = routes
    rss = {}    # depots or rss locations
    weight = input("optimization weight")
    max_duration = input(" Maximum tour duration:")
    total_cap = input("Maximum vehicle capacity")
    n = int (input ("Number of pods: "))
    new_route = []
    first 
    #assign locations with highest distance
    rt1 = {"pod1": 2, "pod2": 5, "pod3": 4}
    rt2 = 0
    while n>0:
        podid = max(rt1,rt2)
        new_route.append (podid)
        if t1 < t2:
            r =
        else:

        k = abs(rt1)
        # distance to visit all pods
"""
 

def distanceList(instance, PODs):
    # Use the distance matrix and find the distances between all the
    # customers in the TSP tour
    # NOTE: this distance list has depot and first customer, but not the last
    # customer back to depot!
    distance = []
    lastCustomerID = 0
    for PodID in instance:
        distance.append(PodID)
        #lastCustomerID = customerID
    return distance

def culmulativeDistance(instance, PODs, startIndex, endIndex):
    # Returns the distance between the start and end index of customers.
    # If the customer is at the beginning or end, includes the depot
    distList = distanceList(instance, PODs)

    if startIndex == 0 or endIndex > len(PODs):
        distance = 9999999999999999999
    elif endIndex < len(PODs):
        distance = sum(distList[startIndex:endIndex+1])
    elif endIndex == len(PODs):
        distance = sum(distList[startIndex:endIndex+1])
        distance += instance['distance_matrix'][PODs[endIndex-1]][0]
    return distance

def culmulativeDemand(instance, PODs, startIndex, endIndex):
    # Returns the total demand of the start and end index of customers.
    dmdList = demandList(instance, PODs)
    demand = sum(dmdList[startIndex:endIndex+1])
    return demand

def demandList(instance, PODs):
    # Use the distance matrix and find the demand of all the
    # customers in the TSP tour
    demand = []
    for customerID in PODs:
        demand.append(instance['PodID_%d' % customerID])
    return demand

def distanceBetweenCustomers(instance, fromCustomer, toCustomer):
    return instance['distance_matrix'][fromCustomer][toCustomer]


def partitionpods(instance, PODs, lightRange=100, lightCapacity=50):
    # The method takes in a TSP tour, the time and capacity constraint
    # Returns a list indicating customers that the light resource
    # is able to deliver to - [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

    considerList = [False] * len(PODs) #zero list with length of PODs
    considerList[0] = True
    considerList[-1] = True
    clusterList = [0] * len(PODs) #zero list with length of PODs

    # Determine the order of the distance list
    distList = distanceList(instance, PODs)
    print (distList)
    sortedDistanceList= [0] * len(distList)
    for i, x in enumerate(sorted(range(len(distList)), key=lambda y: distList[y])):
        sortedDistanceList[x] = i

    # Start the cluster with the closest pair and add neighbouring customers until
    # the range or capacity constraint is reached. Then find the next closest pair
    # not part of any cluster and repeat until all customers are considered
    for i in range(len(PODs)):
        considerCustomer = sortedDistanceList.index(i)
        print ("consider customer index: %d" % considerCustomer)
        # Determine the neighbouring nodes of the considerCustomer
        # Calculate the distance (include rendezvous)
        if considerCustomer == 0:
            clusterEdgeLocation = [considerCustomer, considerCustomer+1]
            distance = 99999999999 #don't consider the first customer as light resource deliverable
        elif considerCustomer == (len(PODs)-1):
            clusterEdgeLocation = [considerCustomer-1, considerCustomer]
            distance = 99999999999 #don't consider the last customer as light resource deliverable
        else:
            clusterEdgeLocation = [considerCustomer-1, considerCustomer+1]
            distance = culmulativeDistance(instance, PODs,
                                        clusterEdgeLocation[0], clusterEdgeLocation[1])

        # Calculate the demand of considerCustomer
        demand = culmulativeDemand(instance, PODs, considerCustomer, considerCustomer)

        # First check if the customer is already considered, range feasibility and demand feasibility
        if (any(considerList[clusterEdgeLocation[0]:clusterEdgeLocation[1]+1]) == True
                or distance > lightRange or demand > lightCapacity):
            continue
        # Passes all tests, initialize considerCustomer as lightCluster
        else:
            considerList[clusterEdgeLocation[0]:clusterEdgeLocation[1]+1] = [True] * (clusterEdgeLocation[1]-clusterEdgeLocation[0]+1)
            clusterList[considerCustomer] = 1

            # Incrementally add the neighbouring customers until the edge of cluster reaches the ends of the list
            # Check range feasibility
            # Check demand feasibility
            while clusterEdgeLocation[0] != 0 or clusterEdgeLocation[1]+1 != len(PODs):
                distanceForward = culmulativeDistance(instance, PODs,
                                                    clusterEdgeLocation[0], clusterEdgeLocation[1]+1)
                distanceBackward = culmulativeDistance(instance, PODs,
                                                    clusterEdgeLocation[0]-1, clusterEdgeLocation[1])
                demandForward =  culmulativeDemand(instance, PODs,
                                                    clusterEdgeLocation[0]+1, clusterEdgeLocation[1])
                demandBackward = culmulativeDemand(instance, PODs,
                                                    clusterEdgeLocation[0], clusterEdgeLocation[1]-1)
                print ("Demand from %d to %d is: %d" % (clusterEdgeLocation[0]+1, clusterEdgeLocation[1], demandForward))
                print ("Demand from %d to %d is: %d" % (clusterEdgeLocation[0], clusterEdgeLocation[1]-1, demandBackward))
                print ("Range is: %d and %d" % (distanceForward, distanceBackward))

                # Greedy approach: look at the shortest distance neighbouring node to add to cluster
                # If neighbouring node successfully pass the demand and time constraint
                # Update the cluster list and the consider list
                # Also check if there is a space for light resource to rendezvous
                if (distanceForward <= distanceBackward and distanceForward < lightRange and demandForward < lightCapacity
                    and clusterList[clusterEdgeLocation[1]+1] == False):
                    considerList[clusterEdgeLocation[0]+1:clusterEdgeLocation[1]+1] = [True] * (clusterEdgeLocation[1]-clusterEdgeLocation[0])
                    clusterList[clusterEdgeLocation[0]+1:clusterEdgeLocation[1]+1] = [1] * (clusterEdgeLocation[1]-clusterEdgeLocation[0])
                    clusterEdgeLocation[1] = clusterEdgeLocation[1] + 1
                    print ("Cluster forwards is added")
                elif (distanceForward > distanceBackward and distanceBackward < lightRange and demandBackward < lightCapacity
                    and clusterList[clusterEdgeLocation[0]-1] == False):
                    considerList[clusterEdgeLocation[0]:clusterEdgeLocation[1]] = [True] * (clusterEdgeLocation[1]-clusterEdgeLocation[0])
                    clusterList[clusterEdgeLocation[0]:clusterEdgeLocation[1]] = [1] * (clusterEdgeLocation[1]-clusterEdgeLocation[0])
                    clusterEdgeLocation[0] = clusterEdgeLocation[0] - 1
                    print ("Cluster backwards is added")
                else:
                    break
                print (clusterList)
                print ("The cluster edge is at: %d and %d" % (clusterEdgeLocation[0], clusterEdgeLocation[1]))
    return clusterList

two_phase()
#getConnections(PODList)
partitionpods(PODList, PODList[0],lightRange=100, lightCapacity=50)
prune_route()