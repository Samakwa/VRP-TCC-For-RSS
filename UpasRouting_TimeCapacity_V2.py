
import numpy as np
import sys
import pkgutil
import ArrayList
import models.Route
import rss_distribution.algorithms.RoutingAlgorithmImpl
import rss_distribution.algorithms.UpasRoutingImpl

# Partition routes based on capacity with the consideration of time

class UpasRouting_TimeCapacity_V2:
    def __init__(self, MaxCapacity, Time,Route):
        self.MaxCapacity = MaxCapacity
        self.Time = Time
        self.Route = Route


#Select route with the largest capacity, unless its under the capacity constraint, then select the route with the max time.
    def selectRouteIndexToOptimize(self,ArrayList, Route, p_allRoutes):
        self.maxCapacityRoute = getMaxCapacityRouteIndex(p_allRoutes)
        if (Route.CurrentRouteCapacity < STANDARD_CAPACITY_ALLOWED_FOR_ROUTE):
            return getMaxTimeRouteIndex(p_allRoutes)
        else:
            return maxCapacityRoute

# Returns the under utilized route to add the next POD in the pool, during re-partitioning
# Select the route with the lowest capacity, unless one is over the time allowed, then select the other.

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

    # Checks if all routes are under both time and capacity constraints.

    def isRoutesUnderConstraints(ArrayList, Route,p_allRoutes):
        if (allRoutesUnderAllowedTime(p_allRoutes) and allRoutesUnderAllowedCapacity(p_allRoutes)):
            return True
        else:
            return False

