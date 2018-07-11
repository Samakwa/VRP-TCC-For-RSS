
import numpy as np
from datetime import date
import csv
import pandas as pd
import math
import sys

import sys
import pkgutil
import logging
#from flask import Flask, render_template, flash, request
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

TIME_ALLOWED_FOR_ROUTE = 180

trimmedRoute = []
podsCutFromRoute =[]
allPods = {}
inPhase1 = True
PODList = {}
dedicatedVehicles = {}
rssLocationIds = {}
routes = {}
route_time = {"r1": 50, "r2": 40}
route_capacity = {"r1r2": 300, "r2r3": 400}
routesUnderConstraints = []
routesCombined = []
AnalyzeMode = []
allowSplitBoxes = []

isPrintPartitioningTest = False

DEFAULT_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = 22
DEFAULT_STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE = 22
DEFAULT_STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE = 9600
DEFAULT_TIME_SPENT_AT_POD = 30.0
DEFAULT_TIME_ALLOWED_FOR_ROUTE = 720.0     #12 hours

STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = 12


def AlgorithmSetup():
    # set up contraints or in analysis mode
    with open('Routes.csv', 'r+') as in_file:
        reader = csv.reader(in_file)
        data_iter = csv.reader(in_file, delimiter=' ')
        data = [data for data in data_iter]
        data_array = np.asarray(data, dtype=None)
        with open('Routes_new.csv', 'w+') as outfile:
            writer = csv.writer(outfile)
            routes = {rows[0]: rows[1] for rows in reader}

    #TIME_ALLOWED_FOR_ROUTE = 180
    USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = input("Enter Number of vehicles to be used in this route ")
    STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE = 20
    STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = 12
    STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE = input("Enter allowed population: ")

    USER_TIME_ALLOWED_FOR_ROUTE = input("Enter Time allowed for this  route:")
    USER_POPULATION_PER_UNIT = input('Enter population: ')

    current_hour = int(input("Number of hours for Route coverage: "))

        # hours allowed * 60 minutes / hour for total time in minutes
    TIME_ALLOWED_FOR_ROUTE = (current_hour * 60.0)

    # if not in analysis mode, but not yet set by user, set form with defaults
    if USER_TIME_ALLOWED_FOR_ROUTE == None:
        USER_TIME_ALLOWED_FOR_ROUTE = TIME_ALLOWED_FOR_ROUTE
            # if not in analysis mode, but using user input
    else:
        # override using user input
        TIME_ALLOWED_FOR_ROUTE = USER_TIME_ALLOWED_FOR_ROUTE

    # do the same for the other parameters
    # if (form.USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE == null):
    if USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE == None:
        USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE * \
                                                       STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE


    else:
            USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE = STANDARD_CAPACITY_ALLOWED_FOR_ROUTE


    if STANDARD_CAPACITY_ALLOWED_FOR_ROUTE == USER_STANDARD_CAPACITY_ALLOWED_FOR_ROUTE:
        USER_UNITS_PER_VEHICLE = input("Number of units per vehicle")
        if (USER_UNITS_PER_VEHICLE == None):
            USER_UNITS_PER_VEHICLE = STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE
        else:
            STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE = USER_UNITS_PER_VEHICLE

    if (USER_POPULATION_PER_UNIT == None):
        USER_POPULATION_PER_UNIT = STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE
    else:
        STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE = USER_POPULATION_PER_UNIT

    USER_TIME_SPENT_AT_POD = input("Enter time to spent on this pod or read from file:")
    TIME_SPENT_AT_POD = input("TIME_SPENT_AT_POD: = ")
    if (USER_TIME_SPENT_AT_POD == None):
            USER_TIME_SPENT_AT_POD = TIME_SPENT_AT_POD
    else:
        TIME_SPENT_AT_POD = USER_TIME_SPENT_AT_POD

        with open('Routes_new.csv', 'w+') as outfile:
            writer = csv.writer(outfile)
            routes = {rows[0]: rows[1] for rows in reader}
    # df = pd.read_sql_query ('SELECT COL1, COL2 FROM Routes where COL2 < 12',RoutingData )



def getPoddistance():
    stepThrough = True
    maxDistance = 0.0
    endingPod = -1
    currentPod = 1
    statusStep = 100.0
    currentsStepStat = 0.0
    Pods = []
    PodsPopn = []
    listOfClosestPods = []

    # get the pods pair distances sorted by shortest distance
    STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE = input("Enter allowed population: ")
    with open("PODslist.csv", "r+") as csvfile:

        visited =[]
        distances ={}
        predecessors = {}
        result = {}
        reader = csv.reader(csvfile)
        #reader = csv.DictReader(csvfile)

        for row in reader:
            #results = dict(reader)
            PODdetails = row[0] + " " + row[2]# + " "+ row[3] +" ,"+ row[4]
            Pods.append(PODdetails)
            PodsPopn.append(row[2])

        maxDistance = int(input ("Enter maximum Distance for this cluster"))
        for PodID in PODList:
            POdID = 1
            # get list of pods ordered by distance
            #implement a Star or Dijkstra
            distance =0
            distance+= distance
            if distance > maxDistance:
                listOfClosestPods.append(PodID)
                PodID += 1

            ClusterPods = []

        #get_distance_pod_pair(r1, r2)
        for item in  listOfClosestPods:
            # x is first POD location in the cluster
            #distance = math.sqrt ((((x[0]-y[0])**2) + ((x[1]-y[1])**2)))
            if (item.distanceToPod > maxDistance):
                maxDistance = listOfClosestPods[item].c_distanceToPod
                startingPod = PodID
                endingPod = listOfClosestPods[item].podId

                print(startingPod + "  to  " + endingPod + "  has a distance of " + maxDistance)
                currentsStepStat += statusStep

            route_time.update()
            print("Analyzing parameters: ")

            currentPod += 1
    # ensure the graph is undirected
    #assume values in PODList dict represent distances
    for i in range(len(PODList)):
        for j in range(len(PODList)):
            if (i != j):
                if PODList.get(i) > PODList.get(j):
                    distanceToPod[i] =   distanceToPod[j]
            # convert pod population to pod demand if not splitting boxes / units
    # If form does not allow split boxes
    for i in range(len(PODList)):
        PODList.get(i).setDemandAsUnits(STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE)

        # check for minimum time allowed set by user with the longest distance
        minTimeError = False
        minimumMaxDistanceFromRSS = -1.0
    firstRSS = True
    RSSList =[]
    TIME_SPENT_AT_POD = input("TIME_SPENT_AT_POD: = ")
    for pod in PODList:
        if pod in RSSList:
            if firstRSS == False:
                # for the max distance from this rss to any pod
                localMax = -1.0
                for PodDistanceNode in listOfClosestPods:
                    if distanceToPod[i]> localMax:
                        localMax = distanceToPod
                    # if this local max is less than the global max, then this
                    # rss must be closer to the pod than the rss that assigned
                    # the global max, so set it to this local max instead
                    if localMax < minimumMaxDistanceFromRSS:
                        minimumMaxDistanceFromRSS = localMax
            elif firstRSS == True:
                for PodDistanceNode in listOfClosestPods:
                    if PODList.distanceToPod > minimumMaxDistanceFromRSS:
                        minimumMaxDistanceFromRSS = PODList.distanceToPod

                firstRSS = False
    minimumMaxDistanceFromRSS += TIME_SPENT_AT_POD
    # remove any demand from pods over capacity, and add dedicated vehicles for these extra demand
    dedicatedVehicles.update(newDemand)
    for pod in PODList:
        # if pod demand is over capacity
        if pod.getDemand > STANDARD_CAPACITY_ALLOWED_FOR_ROUTE:
            # remove the demand that will be used in the dedicated truck where
            # new demand =((demand - (demand % cap)) / cap)
            originalDemand = pod.getDemand()
            newDemand = (originalDemand % STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)
    # check for case where originalDemand was a multiple of STANDARD_CAPACITY_ALLOWED_FOR_ROUTE
    # (mostly common when dealing with capacity in units)

    if (newDemand == 0):
        newDemand = STANDARD_CAPACITY_ALLOWED_FOR_ROUTE

        # if form does not allowSplitBoxes)
        # demand = units here
        # need to calculate new population from the capacity ( in units) times the population each unit can hold
        newPopulation = newDemand * STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE

    else:
        # demand = population here, so no further calculation is needed
        newPopulation = newDemand

    # set the new population (which also calculates a new demand)
    pod.update(newPopulation)
    # save reference of pod id and how many fully loaded dedicated trucks that are needed for the remaining demand
    # dedicated trucks = ((demand - (demand % cap)) / cap)
    dedicatedTrucks = ((originalDemand - newDemand) / STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)

    dedicatedVehicles.get(pod_id)
    dedicatedVehicles.update({pod_id, dedicatedTrucks})

    # ArrayList < POD > originalCopy = new ArrayList < POD > ();
    # copy of pod list  for multiple run of algorithm in analysis mode
    # (creating a deep copy of list)
    allPods.clear()
    rssLocationIds.clear()
    routes.clear()


    for pod in PODList:
        PODListCopy.append(pod.copy)
        # also add to all pods map
        Pods.put([pod_id, pod.copy])
        # and the rss location references
        if pod in rss:
            rssLocationIds.update([pod, pod_id])
        # originalCopy.add(pod.copy)

    loadingdots = 1
    # c_form.updateStatus("computing rss routes...");

    # repeat if needed for analysis variables are reset at the bottom
    while (True):
        # print("---- PARAMS TO RUN WITH ---- ")
        # print"TIME_ALLOWED_FOR_ROUTE: " + TIME_ALLOWED_FOR_ROUTE)
        # print("STANDARD_CAPACITY_ALLOWED_FOR_ROUTE: " + STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)
        # print("STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE: " + STANDARD_CAPACITY_POP_PER_UNIT_ALLOWED_FOR_ROUTE)
        # print("STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE: " + STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE)
        # print("TIME_SPENT_AT_POD: " + TIME_SPENT_AT_POD);
        # print("p_PODList size: " + p_PODList.size())
        # print("PODListCopy size: " + PODListCopy.size())

        # check for minimum time requirement errors
        if (minimumMaxDistanceFromRSS > TIME_ALLOWED_FOR_ROUTE):
            minTimeError = True
            if (minTimeError):
                # if (c_form.inAnalyzeMode()) {
                while (currentHour >= minHours):
                    analysisInfo.addHourResult(analysisInfo.currentHour, -1)
                    currentHour -= 1


            else:
                print("NEED TIME >= " + minimumMaxDistanceFromRSS)

            # reset variables
            routes.clear()
            allPods.clear()
            rssLocationIds.clear()
            PODList.clear()
            break

    RouteName = input("Enter route name: ")
    initRoutes = {"RouteName": [], "Distance": []}
    initRoutes[RouteName].append([startingPod, endingPod, PODList])

    # partition all routes down
    while routeid not in routesUnderConstraints:
        # partition the longest route
        # 2Step_TimeCapacity_V1.selectRouteIndexToOptimize()
        routeOptimizeIndex = selectRouteIndexToOptimize(routes)
        routeToPartition = routes.get(routeOptimizeIndex)
        # ArrayList < Integer > startingEndingPodIds = routeToPartition.getFurthestPodsInRoute()
        Route[newRoutes] = (
        routeToPartition, startingEndingPodIds(0), startingEndingPodIds.get(1), routeToPartition.getRoute())
    # update, add and display new routes
    updateRouteAtIndexWithDisplayUpdate(p_RoutesKML, newRoutes.get(0), routeOptimizeIndex)
    addRouteWithDisplayUpdate(p_RoutesKML, newRoutes.get(1))

    print("computing rss routes" + sdots)
    Routes.update(routes)

    if loadingdots == 5:
        loadingdots = 1
    else:
        loadingdots += 1

    for route in Route:
        print(
            route.getRouteName() + ": (TIME=" + route.getCurrentRouteTime() + ")  --  (CAPACITY=" + route.getCurrentRouteCapacity() + ")")
    # Optimizing Routes
    print("optimizing routes...")
    super_routes = {"r1": 100, "r2": 200, "r3": 300}
    route = input("Enter route: ")
    # combine routes until it is no longer able to retrieve a feasible combination solution

    new_route = super_routes[route].__add__()

    returnOpt = False
    if (returnOpt):

        # dynamic params
        allowRssReturns = False  # allowing route to return / reload at rss
        alpha = 0.85  # reducing factor for time benefit vector
        alpha_initial = 0.85
        x = 0.05  # value to decrease alpha by each iteration when needed
        beta = 1  # scaling the weight

        while (True):
            # dynamic programming route combination optimization
            dynamicAlgo = Dynamic_0_1_knapsack_algorithm(routes, STANDARD_CAPACITY_ALLOWED_FOR_ROUTE,
                                                         TIME_ALLOWED_FOR_ROUTE, rssLocationIds, allPods, alpha, beta,
                                                         allowRssReturns)
            # get the best route combination from the list
            New_super_route = dynamicAlgo.run()

            # if a single route was optimal with the provided params
            # or if the super route is over time
            if (superRoute == None) or (superRoute.getCurrentRouteTime() > TIME_ALLOWED_FOR_ROUTE):
                # lower alpha
                alpha = alpha - x

                if alpha <= 0:
                    if (allowRssReturns):
                        print("Breaks - 01")
                        print(alpha)
                        break

                    # run again, but now allow returns to rss
                    allowRssReturns = True
                    alpha = alpha_initial
                    beta = 2
                continue

            params = params.update
            params[0] = alpha
            params[1] = beta
            if (allowRssReturns):
                params[2] = 1.0
            else:
                params[2] = 0.0

            allParamSave.append(params)
            # add the super route
            super_routes.update(super_route)

            # reset alpha
            alpha = alpha_initial

            # remove the route it combined
            for route in routes:
                if routename in super_routes:
                    routescombined.append(routename)
            for route in routesCombined:
                routes.remove(routename)

            # There should be at least two if there are not at least two routes to try and combine

            if c_routes.size < 2:
                print("Breaks - 02")

            print("\n")
            print("-----ROUTES NOT COMBINED(" + c_routes.size() + "):------\n")

            for route in routes:
                print(routename.getRouteInfoasCSV)

            print("\n")
            print("-----FINAL SUPER ROUTES(" + superRoutes.size() + "):------\n")
            for super_route in superRoutes:
                print(routename.getRouteInfoAsCSV)
            print("\n")
            print("-----FINAL SUPER ROUTE SEGMENTS:------\n")
            print(SuperRoute.getSuperRouteSegmentsInfoCsvHeader)
            for super_route in superRoutes:
                print(routename.getSuperRouteSegmentsInfoAsCSV)

            print("\n\n")
            print("-----PARAMS------\n")
            print("ROUTE_ID, alpha, beta, returns")
            for key in allParamSave.keySet:
                print(key + ", " + allParamSave.get(key)[0] + ", " + allParamSave.get(key)[1] + ", " +
                      allParamSave.get(key)[2] + ", ")

            system.exit

        if (rss in optimizeRoutes) and (rss not in AnalyzeMode):
            # simulated annealing optimize
            optimizedRouteOrder = getOptimizeRouteOrderWithSimulatedAnnealing(routename)
            for i in range(routes.size):
                # did not optimize on less than 4 vertices
                if route.size < 4:
                    continue

                optimizedRoute = configureRouteWithNewOrderOfIds(routename, optimizedRouteOrder, TIME_SPENT_AT_POD)

                if CurrentRouteTime() < c_routes.get(i).getCurrentRouteTime:
                    routename.set(i, optimizedRoute)

                else:
                    print("Keeping original route:\t" + routename)

            c_form.updateRoutes(c_routes)


def reset_route():
    PODListCopy = []
    if rss in AnalyzeMode:
        print("Finished analyzing for hour: ", currenthour + " || ")
        print("Total: " + (c_routes.size() + c_form.getNumberOfExtraVehiclesRequiredFromParams()))
        print(
            "Number Of Routes: " + c_routes.size() + "Number of Extra Routes Adding: " + c_form.getNumberOfExtraVehiclesRequiredFromParams() + ")")

        # add result from analysis
        route_details.append([currentHour], [extratime], [ExtravehicleRequired])

        # check if analysis are complete
        # c_form.analysisInfo
    if (currentHour > minHours):
        # c_form.analysisInfo.currentHour -= 1
        currentHour -= 1

        # update constraints for next run
        # hours allowed * 60 minutes / hour for total time in minutes
        TIME_ALLOWED_FOR_ROUTE = currentHour * 60.0

        # reset routes
        c_routes.clear()

        #   reset pod lists and the map (replace with the copy)
        p_PODList.clear()
        c_allPods.clear()
        c_rssLocationIds.clear()

        for pod in PODListCopy:
            newPOD = pod.copy()

            p_PODList.append(newPOD)
            c_allPods.put(newPOD.c_id, newPOD)

            if newPOD in RSS:
                c_rssLocationIds.add(newPOD.c_id)

            resetRouteNameAndColor()
            # run again
            continue

    # analysis are complete
    # reset variables
    routes.clear()
    allPods.clear()
    rssLocationIds.clear()
    PODList.clear()

    resetRouteNameAndColor()

    # reset to time input, either from user or defaults
    # if not yet set by user, set form with defaults
    if USER_TIME_ALLOWED_FOR_ROUTE == none:
        USER_TIME_ALLOWED_FOR_ROUTE = DEFAULT_TIME_ALLOWED_FOR_ROUTE
    # if not in analysis mode, but using user input
    else:
        # override using user input

        TIME_ALLOWED_FOR_ROUTE = USER_TIME_ALLOWED_FOR_ROUTE


def dedicated_vehicle():
    minTimeError = False
    # add dedicated vehicles for pods whose capacity was reduced
    if rss in AnalyzeMode:
        Route.dedicatedRoutes = input("Enter new route: ")

        for podID in dedicatedVehicles:
            # create a new route at full capacity for demand removed from the algorithm
            id = int(podID)
            numDedicatedRouteForID = dedicatedVehicles.get(podID)
            for i in range(1, numDedicatedRouteForID, 1):
                dedicatedRoute = Route(getRouteName() + "_Dedicated_For_POD__" + id,
                                       STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)

                # set duplicatedPod demand at a demand equalt to full capacity
                newPopulation = STANDARD_CAPACITY_ALLOWED_FOR_ROUTE * STANDARD_CAPACITY_UNITS_ALLOWED_FOR_ROUTE
                duplicatedPod = c_allPods.get(id).copyWithNewPopulation(newPopulation)
                duplicatedPod.c_name = "(dedicated) " + duplicatedPod.c_name

                dedicatedRoute.addPodToEndOfRoute(duplicatedPod, 0.0, TIME_SPENT_AT_POD)
                dedicatedRoute = addInRssLocation(dedicatedRoute)
                dedicatedRoutes.append(dedicatedRoute)
        # add the routes to the display
        for route in dedicatedRoutes:
            RouteWithDisplayUpdate(route)
            # c_form.updateRoutes(c_routes)
    if (minTimeError) and (rss in AnalyzeMode):
        setAllowedTimeError = minimumMaxDistanceFromRSS
    else:
        setFinalResults(c_routes)


"""
def Route selectRouteFromConstraintsToAddPod(Route r1, Route r2) {
throw new UnsupportedOperationException("Needs to be implemented for the specific optimization strategy");

def selectRouteIndexToOptimize(ArrayList < Route > p_allRoutes) {
throw new UnsupportedOperationException("Needs to be implemented for the specific optimization strategy");
}
def routesUnderConstraints(ArrayList < Route > p_allRoutes) {
throw new UnsupportedOperationException("Needs to be implemented for the specific optimization strategy");
"""

"""
Given a two furthest points and a set of points between those two points,
* first if either of the points is not the rss location, remove the rss location from the
* set of points between them.Then equally partition the set of points by choosing
* the closest points to the starting points given.At the end, for each of the new routes
* get the starting and ending point, if both of the starting or ending points are not the
* rss location, then add it to the closest end point to it.
"""


def partitionRoute():
    with open('routes.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            routes.update(row)
    # create two new routes
    r1 = input("enter route name:")
    r2 = input("enter route name:")
    route1 = routes[r1]  # new_Route(p_routeName, STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)
    route2 = routes[r2]  # new_Route(getRouteName(), STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)

    if rssLocationId in startingPod:
        route1.append([c_allPods.get(p_startingPod), 0.0, 0.0])

    else:
        route1.append([c_allPods.get(p_startingPod), 0.0, TIME_SPENT_AT_POD])
        # removePodByIdOnlyFromListOfPods(p_podsToRoute, p_startingPod)
        PODList.remove([p_podsToRoute, p_startingPod])

    if c_rssLocationIds in p_endingPod:
        route2.append([c_allPods.get(p_endingPod), 0.0, 0.0])
    else:
        # addPodToEndOfRoute
        route2.append([c_allPods.get(p_endingPod), 0.0, TIME_SPENT_AT_POD])
        p_podsToRoute = removePodByIdOnlyFromListOfPods(p_podsToRoute, p_endingPod)

    # remove all rss points from list of pods to route
    for rssPod in rssLocationIds:
        p_podsToRoute = removePodByIdOnlyFromListOfPods(p_podsToRoute, rssPod)

    # while there are pods to place into a route
    while p_podsToRoute.size() > 0:
        # select the route with the least constrained constraints to add pod to
        selectRouteFromConstraintsToAddPod()
        updateRoute = selectRouteFromConstraintsToAddPod(route1, route2)

        # get the pod to add
        for pod in c_listOfClosestPods:
            # if the pod closest to it is in the pods route
            if c_podId in PODListWithId:
                # add it to the route
                routes.append([c_allPods.get(pod.c_podId), pod.c_distanceToPod, TIME_SPENT_AT_POD])
                # remove it from the list of pods to route
                # p_podsToRoute.remove(c_allPods.get(pod.c_podId))
                routes = set(routes)
                routes.remove([p_podsToRoute, pod.c_podId])
                break

    # Add back in a point of location (rss) if needed
    if (Location_route1 not in StartingPod) and (Location_route1 not in EndingPod):
        # route1 = addInRssLocation(route1)
        rssLocation.append(route1)
    if (Location_route2 not in StartingPod) and (Location_route2 not in EndingPod):
        rssLocation.append(route2)

    # reverse route to start at the rss if needed
    if Location_route1 not in StartingPod:
        Collections.reverse([route1.getRoute()])
    if Location_route2 not in StartingPod:
        Collections.reverse(route2.getRoute())

    # Return the two new routes
    newRoutes.append(route1)
    newRoutes.append(route2)

    return newRoutes


def removePodByIdOnlyFromListOfPods(listToRemoveFrom, podIdToRemove):
    podToRemove = none
    for pod in listToRemoveFrom:
        if pod.c_id == podIdToRemove:
            podToRemove = pod
            break

    if podToRemove != None:
        listToRemoveFrom.remove(podToRemove)
    return listToRemoveFrom


def PODListContainPodWithId(PODListToCheck, intpodId):
    for pod in PODListToCheck:
        if pod.c_id == podId:
            return True
        else:
            return False


def routeWithLowestTime(r1, r2):
    if (r1.getCurrentRouteTime() < r2.getCurrentRouteTime()):
        return r1
    else:
        return r2


def routeWithLowestCapacity(r1, r2):
    if (r1.getCurrentRouteCapacity() <= r2.getCurrentRouteCapacity()):
        return r1
    else:
        return r2


def addInRssLocation(p_route):
    # get the closest rss points to both the starting pod and the ending pod
    closestRssToStartingPod = findClosestRssPointToPod(p_route.getStartingPod())
    closestRssToEndingPod = findClosestRssPointToPod(p_route.getEndingPod())

    # get the distance from each to compare
    startingPodToRss = [getStartingPod().getDistanceToPodId(closestRssToStartingPod)]
    endingPodToRss = [getEndingPod().getDistanceToPodId(closestRssToEndingPod)]

    if (startingPodToRss <= endingPodToRss):
        # add the rss to the starting point of the route
        p_route.addPodToStartOfRoute(c_allPods.get(closestRssToStartingPod), startingPodToRss, 0.0)
    else:
        # add the rss to the ending point of the route
        p_route.addPodToEndOfRoute(allPods.get(closestRssToEndingPod), endingPodToRss, 0.0)

    return p_route


def findClosestRssPointToPod(pod):
    closestRssId = None
    distance = MAX_VALUE
    for rssId in rssLocationIds:
        if DistanceToPodId(rssId) < distance:
            closestRssId = rssId
            distance = pod.getDistanceToPodId(rssId)
    return closestRssId


def getMaxCapacityRouteIndex(p_allRoutes):
    # set the first one in the list
    maxCapacity = p_allRoutes.get(routeIndex).getCurrentRouteCapacity()

    for i in (0, p_allRoutes.size, 1):
        if ((p_allRoutes.get(i).getCurrentRouteCapacity() > maxCapacity) and (
                p_allRoutes.get(i).getRoute().size() > 2)):
            maxCapacity = p_allRoutes.get(i).getCurrentRouteCapacity()
            routeIndex = i
    return routeIndex


def getMaxTimeRouteIndex(p_allRoutes):
    # set the first one in the list
    routeIndex = 0
    maxRouteTime = p_allRoutes.get(routeIndex).getCurrentRouteTime()

    for i in (0, p_allRoutes.size, 1):
        if ((p_allRoutes.get(i).getCurrentRouteTime() > maxRouteTime) and (p_allRoutes.get(i).getRoute().size() > 2)):
            maxCapacity = p_allRoutes.get(i).getCurrentRouteCapacity()
            routeIndex = i

    return routeIndex


def allRoutesUnderAllowedTime(p_allRoutes):
    for route in p_allRoutes:
        if ((route.getCurrentRouteTime() > TIME_ALLOWED_FOR_ROUTE) and (route.getRoute().size() > 2)):
            return False
        else:
            return True


def allRoutesUnderAllowedCapacity(p_allRoutes):
    for route in p_allRoutes:
        if ((route.getCurrentRouteCapacity() > STANDARD_CAPACITY_ALLOWED_FOR_ROUTE) and (route.getRoute().size() > 2)):
            return False
        else:
            return True


"""
addRouteWithDisplayUpdate(KMLFileWriter p_RoutesKML, Route, p_route) throws
ParserConfigurationException, SAXException, IOException, TransformerException)
c_routes.add(p_route);
    if (!c_form.inAnalyzeMode()) {
    p_RoutesKML.addPlacemarkLineForIDToKMLFile(
    getNextRouteColor(),
    p_route.getRouteName(),
    p_route.getRoute()

updateRouteAtIndexWithDisplayUpdate(KMLFileWriter
p_RoutesKML, Route
p_route, int
p_index) throws
ParserConfigurationException, SAXException, IOException, TransformerException
{
c_routes.set(p_index, p_route);

if (!c_form.inAnalyzeMode()) {
p_RoutesKML.updateKMLFilePlacemarkLineForID(
KMLFileWriter.LINE_STYLE.BLUE, // doenst consider new color here yet...
p_route.getRouteName(),
p_route.getRoute()
"""


def printPodDiffsFromLists(l1, l2):
    diff = False
    listSize = l1.size()
    if (l2.size() > listSize):
        diff = True
        listSize = l2.size()

    for i in range(0, listSize, 1):
        p1 = l1.get(i)
        p2 = l2.get(i)

        if (p1.c_id != p2.c_id) or (p1.c_name != p2.c_name) or (p1.c_lon != p2.c_lon) or (p1.c_lat != p2.c_lat) or \
                (p1.getPopulation() != p2.getPopulation()) or (p1.getDemand() != p2.getDemand()) or (
                p1.c_street != p2.c_street) or \
                (p1.c_city != p2.c_city) or (p1.c_listOfClosestPods != p2.c_listOfClosestPods):
            diff = True
            print("\n")
            print("---- PODs ARE DIFFERENT -------")
            print("POD1 \t POD2")

            print(p1.c_id + " \t " + p2.c_id)
            print(p1.c_name + " \t " + p2.c_name)
            print(p1.c_lon + " \t " + p2.c_lon)
            print(p1.c_lat + " \t " + p2.c_lat)
            print(p1.getPopulation() + " \t " + p2.getPopulation())
            print(p1.getDemand() + " \t " + p2.getDemand())
            print(p1.c_street + " \t " + p2.c_street)
            print(p1.c_city + " \t " + p2.c_city)
            print(p1.c_listOfClosestPods + " \t " + p2.c_listOfClosestPods)

            print("-------------------------------")

    return diff


def selectRouteFromConstraintsToAddPod(r1, r2):
    if (r1.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE and
            r2.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):

        return routeWithLowestCapacity(r1, r2)
    # if only one is over the time allowed, return the other
    elif (r1.getCurrentRouteTime() >= TIME_ALLOWED_FOR_ROUTE and
          r2.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):
        return r2

    elif (r2.getCurrentRouteTime() >= TIME_ALLOWED_FOR_ROUTE and
          r1.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):
        return r1

    else:
        return routeWithLowestCapacity(r1, r2)

def get_distance_customers_pair(c1: Pod1D, c2: PodID) -> float:
    return math.hypot(c2.x - c1.x, c2.y - c1.y)
"""
#AlgorithmSetup()
getPoddistance()

demand_redistribution()
reset_route()
dedicated_vehicle()
partitionRoute()
removePodByIdOnlyFromListOfPods()
podListContainPodWithId()
routeWithLowestTime()
routeWithLowestCapacity()
addInRssLocation()
findClosestRssPointToPod()
getMaxCapacityRouteIndex()
getMaxTimeRouteIndex()
isallRoutesUnderAllowedTime()
isallRoutesUnderAllowedCapacity()
printPodDiffsFromLists()

selectRouteFromConstraintsToAddPod()
isRoutesUnderConstraints()
selectRouteIndexToOptimize()
setupRoutesForSecondPhase()
get_distance_customers_pair()

"""
partitionRoute()