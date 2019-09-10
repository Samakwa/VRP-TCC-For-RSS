from rss.DAOs import rssDAO
from flask import jsonify
from decimal import Decimal
import json
import sys
import math

class rssController:
    def __init__(self, workingCopyName):
        self.routes = []
        self.RSSDAO = rssDAO.rssDAO(workingCopyName)
        self.RSSFacilities = []

    def runRSS(self, pods, hour, rsses):
        #loads pod data and RSS data
        pods = json.loads(pods)
        rsses = json.loads(rsses)

        #adds rss facilities to rss controller
        for f in rsses:
            self.RSSFacilities.append(f)

        #print("TEST: ")
        #print(rsses)
        #print("TEST OVER")

        #gets pods farthest from each other
        #farthest = self.findFarthestRoute(pods)
        distance = self.euclideanDistance(pods)

        rssdata = self.routeStarter(distance[0],distance[1],pods,int(hour))
        #rssdata = self.calcRoutesTime(distance[0],distance[1],pods,int(hour))
        rssdata = self.calcRoutesCapacity(distance[0],rssdata,hour)

        print("\n RSS for:" + hour + '\n' + "rssdata:" + '\n' + str(rssdata[0]) + '\n' + str(rssdata[1]) + '\n')

        return rssdata

    #this function tries to make one route with all pods
    def routeStarter(self, adjacencyList, starters, pods,hour):

        #makes the beginning of route [[listofpods],distance,amount,time]
        route = [[starters[0]],0,starters[0]['population'],0]

        #makes a list of available pods
        listOfPods = []
        for pod in pods:
                if(pod != starters[0]):
                    listOfPods.append(pod)

        #while there are pods to choose
        while listOfPods:

            #gets last pod in route
            currentPODId = route[0][-1]

            #finds closest unused pod to current pod
            closePODToCurrentPOD = self.closestpod(currentPODId['podId'],adjacencyList, listOfPods)

            #pod added to route
            route[0].append(closePODToCurrentPOD[0])

            #distance added to route
            route[1] += closePODToCurrentPOD[1]

            #capacity added to route
            route[2] += int(closePODToCurrentPOD[0]['population'])

            #time added to route
            route[3] += 30/60 #driving time
            route[3] += (int(closePODToCurrentPOD[0]['loadingTime'])/60)

            #pod removed from available list
            listOfPods.remove(closePODToCurrentPOD[0])

        #now we need to clip to rss
        route = self.clipToRSS(route)

        print("ROUTE DATA: ")
        running = ""
        for pod in route[0]:
            if('facility_id' in pod):
                running += str(pod['facility_id']) + ", "
            else:
                running += str(pod['podId']) + ", "

        print("ROUTE PATH: ",running)
        print("DISTANCE: ",route[1])
        print("POPULATION SERVED: ", route[2])
        print("TIME TO COMPLETE: ", route[3])

        #if the route exceeds the time alloted
        if(route[3] > hour):

            #breaks the route into smaller routes
            route = self.calcRoutesTime(adjacencyList, starters, pods,hour)

        #returns a route, which could contain more routes. (looks like  [[listofpods],distance,amount,time] or [route1,route2])
        return route

    #this function makes two or more routes
    def calcRoutesTime(self, adjacencyList, starters, pods, hour):

        #makes beginning routes [[listofpods],distance,amount,time]
        route1 = [[starters[0]],0,starters[0]['population'],0]
        route2 = [[starters[1]],0,starters[1]['population'],0]

        #makes a list of available pods
        listOfPods = []
        for pod in pods:
            if(pod != starters[0] and pod != starters[1]):
                listOfPods.append(pod)

        #while there are pods to choose
        while listOfPods:

            # route1 distance <= route2 distance
            if(route1[1] <= route2[1]):

                #gets last pod in route
                currentPODId = route1[0][-1]

                #finds closest unused pod to current pod
                closePODToCurrentPOD = self.closestpod(currentPODId['podId'],adjacencyList, listOfPods)

                #pod added to route
                route1[0].append(closePODToCurrentPOD[0])

                #distance added to route
                route1[1] += closePODToCurrentPOD[1]

                #capacity added to route
                route1[2] += int(closePODToCurrentPOD[0]['population'])

                #time added to route
                route1[3] += 30/60 #driving time
                route1[3] += (int(closePODToCurrentPOD[0]['loadingTime'])/60)

                #pod removed from available list
                listOfPods.remove(closePODToCurrentPOD[0])

            # route1 distance > route2 distance
            else:

                #gets last pod in route
                currentPODId = route2[0][-1]

                #finds closest unused pod to current pod
                closePODToCurrentPOD = self.closestpod(currentPODId['podId'],adjacencyList, listOfPods)

                #pod added to route
                route2[0].append(closePODToCurrentPOD[0])

                #distance added to route
                route2[1] += closePODToCurrentPOD[1]

                #capacity added to route
                route2[2] += int(closePODToCurrentPOD[0]['population'])

                #time added to route
                route2[3] += 30/60 #driving time
                route2[3] += (int(closePODToCurrentPOD[0]['loadingTime'])/60)

                #pod removed from available list
                listOfPods.remove(closePODToCurrentPOD[0])

        #now we need to clip to rss for both routes
        route1 = self.clipToRSS(route1)
        route2 = self.clipToRSS(route2)

        #outputs the routes for testing
        print("\n\nROUTE1 DATA: ")
        running = ""
        for pod in route1[0]:
            if('facility_id' in pod):
                running += str(pod['facility_id']) + ", "
            else:
                running += str(pod['podId']) + ", "

        print("ROUTE PATH: ",running)
        print("DISTANCE: ",route1[1])
        print("POPULATION SERVED: ", route1[2])
        print("TIME TO COMPLETE: ", route1[3])

        print("\n\nROUTE2 DATA: ")
        running = ""
        for pod in route2[0]:
            if('facility_id' in pod):
                running += str(pod['facility_id']) + ", "
            else:
                running += str(pod['podId']) + ", "

        print("ROUTE PATH: ",running)
        print("DISTANCE: ",route2[1])
        print("POPULATION SERVED: ", route2[2])
        print("TIME TO COMPLETE: ", route2[3])

        #if route1's time to complete is over the specified amount.
        if(route1[3] > hour):
            templistOfPods = []

            #get the pods in route1
            for pod in route1[0]:
                if('podId' in pod):
                    templistOfPods.append(pod)

            #gets the adjacencyList and the starters for only pods in route1
            passins = self.euclideanDistance(templistOfPods)

            #splits the route into 2 routes
            route1 = self.calcRoutesTime(passins[0],passins[1],templistOfPods,int(hour))

        #if route2's time to complete is over the specified amount.
        if(route2[3] > hour):
            templistOfPods = []

            #get the pods in route2
            for pod in route2[0]:
                if('podId' in pod):
                    templistOfPods.append(pod)

            #gets the adjacencyList and the starters for only pods in route1
            passins = self.euclideanDistance(templistOfPods)

            #splits the routes
            route2 = self.calcRoutesTime(passins[0],passins[1],templistOfPods,int(hour))

        #returns routes that are time compliant
        return [route1,route2]

    def calcRoutesCapacity(self, adjacencyList, routes, hour):
        if(len(routes[0]) == 2):
            route1 = self.calcRoutesCapacity(routes[0])

        if(len(routes[1]) == 2):
            route2 = self.calcRoutesCapacity(routes[1])

        #18-wheeler assumption
        pallets = 22
        populationServedPerPallets = 9600
        totalServable = pallets * populationServedPerPallets

        print("ROUTES:",routes)

        listOfPods = []

        #routes = [[listofpods],distance,amount,time]
        #pod = is a dictionary
        if(routes[0][2] > totalServable):
            print("\nBEFORE: ")
            running = ""
            for pod in routes[0][0]:
                if('facility_id' in pod):
                    running += str(pod['facility_id']) + ", "
                else:
                    running += str(pod['podId']) + ", "

            print("ROUTE PATH: ",running)
            print("DISTANCE: ",routes[0][1])
            print("POPULATION SERVED: ", routes[0][2])
            print("TIME TO COMPLETE: ", routes[0][3])

            while(routes[0][2] > totalServable):
                if 'facility_id' in routes[0][0][-1]:

                    #removes population from route
                    routes[0][2] -= routes[0][0][-2]['population']

                    #remove time to travel from route
                    routes[0][3] -= int(routes[0][0][-2]['loadingTime'])/60
                    routes[0][3] -= 30/60 #driving time

                    #remove distance from route using the adjacencyList
                    for pod1 in adjacencyList:
                        if pod1[0] == routes[0][0][-2]['podId']:
                            for podDistancePair in pod1[1]:
                                if podDistancePair[0] == routes[0][0][-3]['podId']:
                                    routes[0][1] -= podDistancePair[1]

                    #adds pod closest to the rss to a list of available pods / removes it from the routes
                    listOfPods.append(routes[0][0][-2])
                    del routes[0][0][-2]

                else:

                    #removes population from route
                    routes[0][2] -= routes[0][0][1]['population']

                    #remove time to travel from route
                    routes[0][3] -= int(routes[0][0][1]['loadingTime'])/60
                    routes[0][3] -= 30/60 #driving time

                    #remove distance from route using the adjacencyList
                    for pod1 in adjacencyList:
                        if pod1[0] == routes[0][0][1]['podId']:
                            for podDistancePair in pod1[1]:
                                if podDistancePair[0] == routes[0][0][2]['podId']:
                                    routes[0][1] -= podDistancePair[1]

                    #adds pod closest to the rss to a list of available pods / removes it from the routes
                    listOfPods.append(routes[0][0][1])
                    del routes[0][0][1]

        print("\nAFTER: ")
        running = ""
        for pod in routes[0][0]:
            if('facility_id' in pod):
                running += str(pod['facility_id']) + ", "
            else:
                running += str(pod['podId']) + ", "

        print("ROUTE PATH: ",running)
        print("DISTANCE: ",routes[0][1])
        print("POPULATION SERVED: ", routes[0][2])
        print("TIME TO COMPLETE: ", routes[0][3])

        if(len(listOfPods) > 0):
            #gets the adjacencyList and the starters for only pods in the list
            passins = self.euclideanDistance(listOfPods)

            #splits the routes
            routes[0] = [routes[0], self.routeStarter(passins[0],passins[1],listOfPods,int(hour))]

            listOfPods = []

        if(routes[1][2] > totalServable):
            print("\nBEFORE: ")
            running = ""
            for pod in routes[1][0]:
                if('facility_id' in pod):
                    running += str(pod['facility_id']) + ", "
                else:
                    running += str(pod['podId']) + ", "

            print("ROUTE PATH: ",running)
            print("DISTANCE: ",routes[1][1])
            print("POPULATION SERVED: ", routes[1][2])
            print("TIME TO COMPLETE: ", routes[1][3])

            while(routes[1][2] > totalServable):
                if 'facility_id' in routes[1][0][-1]:

                    #removes population from route
                    routes[1][2] -= routes[1][0][-2]['population']

                    #remove time to travel from route
                    routes[1][3] -= int(routes[1][0][-2]['loadingTime'])/60
                    routes[1][3] -= 30/60 #driving time

                    #remove distance from route using the adjacencyList
                    for pod1 in adjacencyList:
                        if pod1[0] == routes[1][0][-2]['podId']:
                            for podDistancePair in pod1[1]:
                                if podDistancePair[0] == routes[1][0][-3]['podId']:
                                    routes[1][1] -= podDistancePair[1]

                    #adds pod closest to the rss to a list of available pods / removes it from the routes
                    listOfPods.append(routes[1][0][-2])
                    del routes[1][0][-2]

                else:

                    #removes population from route
                    routes[1][2] -= routes[1][0][1]['population']

                    #remove time to travel from route
                    routes[1][3] -= int(routes[1][0][1]['loadingTime'])/60
                    routes[1][3] -= 30/60 #driving time

                    #remove distance from route using the adjacencyList
                    for pod1 in adjacencyList:
                        if pod1[0] == routes[1][0][1]['podId']:
                            for podDistancePair in pod1[1]:
                                if podDistancePair[0] == routes[1][0][2]['podId']:
                                    print("\n\nTESTER: ",podDistancePair[1])
                                    routes[1][1] -= podDistancePair[1]

                    #adds pod closest to the rss to a list of available pods / removes it from the routes
                    listOfPods.append(routes[1][0][1])
                    del routes[1][0][1]

        print("\nAFTER: ")
        running = ""
        for pod in routes[1][0]:
            if('facility_id' in pod):
                running += str(pod['facility_id']) + ", "
            else:
                running += str(pod['podId']) + ", "

        print("ROUTE PATH: ",running)
        print("DISTANCE: ",routes[1][1])
        print("POPULATION SERVED: ", routes[1][2])
        print("TIME TO COMPLETE: ", routes[1][3])

        if(len(listOfPods) > 0):
            #gets the adjacencyList and the starters for only pods in the list
            passins = self.euclideanDistance(listOfPods)

            #splits the routes
            routes[1] = [routes[1], self.routeStarter(passins[0],passins[1],listOfPods,int(hour))]

            listOfPods = []

        return routes

    def findFarthestRoute(self, pods):
        temp = ''
        for pod in pods:
            temp += (pod['name']) + ', '
        return temp

    def mapPods(self):
        #makes verticies for the road map
        #verts = self.RSSDAO.makeVert()
        #print("TESTER: ", verts)

        #if there are verticies, map roads to pods
        #if( verts == 'Successful'):
        #    print("test")
        #return verts
        return 'yay'

    #gets the euclidean distance for all the pods and returns the largest
    def euclideanDistance(self, pods):
        adjacencyList  = []
        farthestList = []
        fardistance = 0

        #for each pod, see if you have a farther distance than the current far distance.
        for pod in pods:

          #list of distances to every node
          listOfDistFromPods = []

          #gets the distance from this node to every other node
          for pod2 in pods:

              #add distance to the adjacencyList list for pod1 -> pod2
              distance = math.sqrt(((float(pod['latitude']) - float(pod2['latitude']))**2) + ((float(pod['longitude']) - float(pod2['longitude']))**2))
              listOfDistFromPods.append([pod2['podId'],distance])

              #make farthest pair if this is a farther distance
              if(distance > fardistance):
                  fardistance = distance
                  farthestList = [pod,pod2,fardistance]

          #adds the pod, and then the corresponding distances
          adjacencyList.append([pod['podId'],listOfDistFromPods])

        #for testing
        #print(adjacencyList)

        #returns [pod1,pod2,distance]
        return [adjacencyList,farthestList]

    #this finds the closest pod to a given pod
    def closestpod(self,podid,adjacencyList, listOfPods):
        closest = ['placeholder',sys.float_info.max]

        #finds the pod with matching id
        for p in adjacencyList:
            if(p[0] == podid):

                #finds the closest pod to the current pod
                for distance in p[1]:
                    if(distance[1] < closest[1] and distance[1] != 0):

                        #looks to see if the podID in in the list of pods
                        for pod in listOfPods:
                            if pod['podId'] == distance[0]:
                                closest = distance
                                break

                #gets pod object for the id found
                for pod in listOfPods:
                    if(pod['podId'] == closest[0]):
                        closest[0] = pod
                        break

        #returns [closest pod, distance]
        return closest

    def clipToRSS(self, route):
        distanceMatrix = []

        #for all rss facilities
        for f in self.RSSFacilities:
            #find the closest distance from the ends of a route.
            distanceFromBeginning = math.sqrt(((float(route[0][0]['latitude']) - float(f['latitude']))**2) + ((float(route[0][0]['longitude']) - float(f['longitude']))**2))
            distanceFromEnd = math.sqrt(((float(route[0][-1]['latitude']) - float(f['latitude']))**2) + ((float(route[0][-1]['longitude']) - float(f['longitude']))**2))

            #add smallest distances to matrix as [ beginning(0)/end(1) , rssId, distance]
            if distanceFromBeginning <= distanceFromEnd:
                distanceMatrix.append([0, f, distanceFromBeginning])
            else:
                distanceMatrix.append([1, f, distanceFromEnd])

        #gets smallest distance from rss and route beginning/end
        smallestPair = ['placeholder','placeholder',sys.float_info.max]
        for distancePair in distanceMatrix:
            if distancePair[-1] < smallestPair[-1]:
                smallestPair = distancePair

        #adds distance and driving time to route [[listofpods],distance,amount,time]
        route[1] += smallestPair[-1]
        route[3] += 30/60 #driving time

        #adds rss to beginning/end of the route
        if smallestPair[0] == 0:
            route[0].insert(0,smallestPair[1])
        else:
            route[0].append(smallestPair[1])

        #returns route with rss attached
        return route
