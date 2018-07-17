
import numpy as np
import sys
import pkgutil

import logging
TIME_ALLOWED_FOR_ROUTE = 180

trimmedRoute = []
podsCutFromRoute =[]
inPhase1 = True
podList =[]


def algorithmSelection():

    pass
    inPhase1 = True


# Returns the under utilized route add the next POD in the pool during re-partitioning.

def selectRouteFromConstraintsToAddPod(r1, r2):
    if (inPhase1):
        # enter condition
        return routeWithLowestTime(r1, r2)

    else:
        #if both under the time constraint, return the lowest capacity.(both under time constraint):
        if (r1.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE and
          r2.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):

            return routeWithLowestCapacity(r1, r2)
        #if only one is over the time allowed, return the other
        elif (r1.getCurrentRouteTime() >= TIME_ALLOWED_FOR_ROUTE and
          r2.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):
            return r2

        elif (r2.getCurrentRouteTime() >= TIME_ALLOWED_FOR_ROUTE and
        r1.getCurrentRouteTime() < TIME_ALLOWED_FOR_ROUTE):
            return r1

        else:
            return routeWithLowestCapacity(r1, r2)
"""
Select route to optimize:
Phase 1: Select the route with the longest time
Phase 2: Select the route with largest current capacity being used.
"""

def selectRouteIndexToOptimize(ArrayList, Route, p_allRoutes):
    if (inPhase1):
        return getMaxTimeRouteIndex(p_allRoutes)
    else:
        maxCapacityRoute = max(p_allRoutes)
        if (p_allRoutes.get(maxCapacityRoute).underCapacity()):
            return getMaxTimeRouteIndex(p_allRoutes)
        else:
            return maxCapacityRoute

#Checks if all routes are under constraints
# Phase 1: Check if all routes are under time
#Phase 2:

def routesUnderConstraints(ArrayList, Route, p_allRoutes):
    inPhase1 = True
    if (inPhase1):
        if (allRoutesUnderAllowedTime(p_allRoutes)):
            # move to the second phase of the algorithm once all routes are under the time constraints
            inPhase1= False

            # first check is any route is over capacity before entering phase 2
            if (allRoutesUnderAllowedCapacity(p_allRoutes)):
                return True

            #setup routes for phase 2
            setupRoutesForSecondPhase()
        return False
    else:
        return (allRoutesUnderAllowedTime(p_allRoutes) and allRoutesUnderAllowedCapacity(p_allRoutes))


def resetIfNeededFromAnalyzeParamsMode():
    inPhase1 = True

def setupRoutesForSecondPhase():
    # collect the new pool of pods removed from routes
    newpodslist =[]

    Array[POD] [podsCutFromRoute] += ArrayList [POD][]


   # for each route
    for route_index in range(0,routes.size):
        route = c_routes.get(route_index)
        print (route_index)
        #if the route is over its capacity contraint
        if (route_Capacity == "over"):
            capacityUsed = 0

            #from the end of the route to the start (rss)
            for i in range (len(routes)):
                # Check if the current route can add another pod capacity
                if ((capacityUsed + route.getRoute.get(i).getDemand())<= route.getAllowedRouteCapacity()):
                    # add it and continue searching  for the POD that pushed it over capacity
                        #capacityUsed += route.getRoute.get.getDemand()

                    #if adding the next pod heading back to the start of the route at index i would be over the capacity,
                    #  cut the route and place the other pods in a new pool
                else:
                    # cut current route to equal  (i+1, i+2, ..n) where 0 is the RSS and n is the last pod in the route
                    # Therefore, this excludes the original RSS
                    Route.trimmedRoute += Route(route.getRouteName(), route.getAllowedRouteCapacity())
                    for j in range ([i+ 1],[route.size],1):
                        #get the pod to add to the trimmed route
                        POD.pod = route.getRoute().get(j)
                        #get the pod travel time cost
                        travelTime =0.0

                        #if not the first pod in the new trimmed route, get the travel time to this pod from the previous pod
                        if (j > i+1):
                            POD.previousPod = route.getRoute().get(j-1)
                            travelTime = pod.getDistanceToPodId(previousPod.c_id)

                        #else add the pod with no travel time until adding back in the rss
                        else:

                            travelTime = 0.0
                        # add the pod to the new trimmed route
                        trimmedRoute.addPodToEndOfRoute(pod, travelTime, TIME_SPENT_AT_POD)
                    #add back in the closest RSS to this new route
                    trimmedRoute = addInRssLocation(trimmedRoute)

                    if  (route_id in trimmedRoute) and (not in rssLocationIds):
                        Collections.reverse(trimmedRoute.getRoute())

                        #update the old route to its new trimmed route

                        trimmedRoute.append([route_index, trimmedRoute])

                        # add remaining pods to the pool of pods cut from routes
                        for  j in range (1,i,1):

                            podsCutFromRoute.append (j)
                    else:
                        break   # Continue to the next route over capacity

    # update the routes that were trimmed with their new trimmed route

    try:
        for  key in routesTrimmed :
            #updateRouteAtIndexWithDisplayUpdate(routesKML_localReference, routesTrimmed.get(key), key)
            routesKML_localReference.update (key)
    except Exception:

        Logging.exception("message")

    if (podsCutFromRoute.size() == 1):
        POD.onlyPod = podsCutFromRoute.get(0)
        Route.route1 = Route(getRouteName(), STANDARD_CAPACITY_ALLOWED_FOR_ROUTE)
        # add-in the only pod and connect it to the rss
        route1.addPodToEndOfRoute(c_allPods.get(onlyPod.c_id), 0.0, TIME_SPENT_AT_POD)
        route1 = addInRssLocation(route1)

        #reverse it if needed
        if route1.getStartingPod in rssLocationIds:
            Collections.reverse(route1.getRoute())

        # add it to list of all routes and update display
        try:
            RouteWithDisplayUpdate.update([localReference, route1])
        except Exception:
            Logging.exception(UpasRouting_2Step_TimeCapacity_V1.error)

    #otherwise, create the 2 new routes from the pool and start phase 2
    else:
        #find the two largest distance between any of the pods in the pool
        maxDistance = 0.0
        POD.routeStartPod = None
        POD.routeEndPod = None
        for POD.startPod in podsCutFromRoute:
            for PodDistanceNode.endPod in startPod.listOfClosestPods:
                if [podsCutFromRoute, endPod.c_podId] in podList:
                    if endPod.c_distanceToPod >= maxDistance:
                        routeStartPod = startPod
                        routeEndPod = c_allPods.get(endPod.c_podId)
                        maxDistance = endPod.c_distanceToPod


        # for a second phase partitioning
        ArrayList[Route][newRoutes] = partitionRoute(getRouteName(), routeStartPod.c_id, routeEndPod.c_id, podsCutFromRoute)
        try:
            # update the displays
            RouteWithDisplayUpdate.append([routesKML_localReference, newRoutes.get(0)])
            RouteWithDisplayUpdate.append([routesKML_localReference, newRoutes.get(1)])
        except Exception:
            Logging.exception(UpasRouting_2Step_TimeCapacity_V1.error)
            # Logger.getLogger(UpasRouting_2Step_TimeCapacity_V1.class.getName()).log(Level.SEVERE, null, ex)


