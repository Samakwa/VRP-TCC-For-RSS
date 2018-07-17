import numpy as np
from datetime import date
import csv
import pandas as pd
import math
import sys

routes = set()
def two_phase():
    with open('Routes.csv', 'r+') as in_file:
        reader = csv.reader(in_file, delimiter='\t')
        PODList = list(reader)

        #with open('Routes_new.csv', 'w+') as outfile:
        #    writer = csv.writer(outfile)
         #   routes = {rows[0]: rows[1] for rows in reader}

        #PODList = []  # contains a list of all locations to be served
        routes = set()
        opt_weight = 1
        reduction_factor= range(0,1)
        maxDistance = int(input("Enter maximum Distance for this cluster"))
        #Assume pod1 is the first location
        ClosestPods = []

        for PodID in PODList:
            POdID = 1
            # get list of pods ordered by distance
            # implement A Star or Dijkstra to get distance
            dijsktra(graph, initial)
            distance += distance
            if distance < maxDistance:
                ClosestPods.append(PodID)
                PodID += 1

        for i,j in range(len(ClosestPods)):
            if POD[i].distance > POD[j]:
                rt1.append(POD[i])
            else:
                rt2.append(POD[j])
            POdID = 1
            # get list of pods ordered by distance
            # implement A Star or Dijkstra to get distance
            distance += distance
            if distance < maxDistance:
                ClosestPods.append(PodID)
                PodID += 1

        totaldistance =1
        #add cumulative distance
        for start, end in listOfClosestPods:
            if start == end:
                break
            else:

                dijsktra(route, start)
                start= end

            totaldistance+= totaldistance

            if totaldistance> maxDistance:
                prune+route()

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
        x_squared = pow((self.x - target.x), 2)
        y_squared = pow((self.y - target.y), 2)

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



def partition_route():
    #inputs
    locationids = routes
    rss = {}    # depots or rss locations
    weight = input("optimization weight")
    max_duration = input(" Maximum tour duration:")
    total_cap = input("Maximum vehicle capacity")
    n = int (input ("Number of pods: "))
    new_route = []
    #assign locations with highesrt distance
    rt1 = {"pod1": 2, "pod2": 5, "pod3": 4}
    rt2 = 0
    while n>0:
        podid = max(rt1,rt2)
        new_route.append (podid)
        if t1 < t2:
            r =
        else:

        k = abs(rt1)




