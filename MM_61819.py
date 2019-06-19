#from utility import FeatureCollectionBuilder as fcb
#from flask import jsonify
from decimal import Decimal
import json
import sys
import math
import pandas as pd

class rssController:
    def __init__(self, workingCopyName):
        self.routes = {}


    def runRSS(self, pods, hour, rsses):
        #loads pod data and RSS data
        df = pd.read_csv('long_lat.csv')
        #pods = json.loads(pods)
        df2 = pd.read_csv('rsses.csv')
        pods = df
        #rsses = json.loads(rsses)
        rsses = df2

        print("TEST: ")
        print(rsses)
        print("TEST OVER")

        #gets pods farthest from each other
        farthest = self.findFarthestRoute(pods)
        distance = self.euclideanDistance(pods)

        rssdata = self.calcRoutes(distance[0],distance[1],pods,int(hour))

        print("\n RSS for:" + hour + '\n' + "rssdata:" + '\n' + str(rssdata[0]) + '\n' + str(rssdata[1]) + '\n')

        return rssdata

    def calcRoutes(self, adjacencyList, starters, pods, hour):
        #18 wheeler assumtion
        unitsPerRoute = 22
        peoplePerUnit = 9600
        totalServable = unitsPerRoute * peoplePerUnit

        #makes beginning routes [[listofpods],distance,amount,time]
        route1 = [[starters[0]['podId']],0,starters[0]['population'],0]
        route2 = [[starters[1]['podId']],0,starters[1]['population'],0]

        #makes a list of available pods
        listOfPods = []
        for pod in pods:
            if(pod['podId'] != starters[0]['podId'] and pod['podId'] != starters[1]['podId']):
                listOfPods.append(pod['podId'])

        while listOfPods:
            # route1 distance <= route2 distance
            if(route1[1] <= route2[1]):
                #we look for the closest pod to the current pod of route1
                current = route1[0][-1]

                #finds the closest pod
                closepod = self.closestpod(current,adjacencyList, listOfPods)

                #finds the pod that is the closest
                foundpod = ''
                for pod in pods:
                    if(pod['podId'] == closepod[0]):
                        foundPod = pod

                #pod id added to route
                route1[0].append(closepod[0])

                #distance added to route
                route1[1] += closepod[1]

                #capacity added to route
                route1[2] += foundPod['population']

                #time added to route
                route1[3] += 30/60 #driving time
                route1[3] += (foundPod['loadingTime']/60)

                listOfPods.remove(closepod[0])

            # route1 distance > route2 distance
            else:
                #we look for the closest pod to the current pod of route1
                current = route2[0][-1]
                closepod = self.closestpod(current,adjacencyList, listOfPods)

                #finds the pod that is the closest
                foundpod = ''
                for pod in pods:
                    if(pod['podId'] == closepod[0]):
                        foundPod = pod

                #pod id added to route
                route2[0].append(closepod[0])

                #distance added to route
                route2[1] += closepod[1]

                #capacity added to route
                route2[2] += foundPod['population']

                #time added to route
                route2[3] += 30/60 #driving time
                route2[3] += (foundPod['loadingTime']/60)

                listOfPods.remove(closepod[0])

        # print("I RAN: ", str(len(pods)))
        # print("ROUTE1: ", route1)
        # print("ROUTE2: ", route2)

        # if(len(route2[0]) == 1):
        #     print("I ONLY HAVE 1")

        if(route1[3] > hour or route1[2] > totalServable):
            #get the pods in route1
            templistOfPods = []
            for entry in adjacencyList:
                if(entry[0] in route1[0]):
                    for pod in pods:
                        if(pod['podId'] == entry[0]):
                            templistOfPods.append(pod)

            #gets the adjacencyList and the starters for only pods in route1
            passins = self.euclideanDistance(templistOfPods)

            #splits the routes
            route1 = self.calcRoutes(passins[0],passins[1],templistOfPods,int(hour))

        if(route2[3] > hour or route2[2] > totalServable):
            #get the pods in route2
            templistOfPods = []
            for entry in adjacencyList:
                if(entry[0] in route2[0]):
                    for pod in pods:
                        if(pod['podId'] == entry[0]):
                            templistOfPods.append(pod)

            #gets the adjacencyList and the starters for only pods in route2
            passins = self.euclideanDistance(templistOfPods)

            #splits the routes
            route2 = self.calcRoutes(passins[0],passins[1],templistOfPods,int(hour))


        return [route1,route2]


    def findFarthestRoute(self, pods):
        temp = ''
        for pod in pods:
            temp += (pod['name']) + ', '
        return temp

    def mapPods(self):
        return 'jobs done'

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
                distance = math.sqrt(((float(pod['latitude']) - float(pod2['latitude']))**2) + ((float(pod['longitude']) - float(pod2['longitude']))**2))
                listOfDistFromPods.append([pod2['podId'],distance])
                if(distance > fardistance):
                    fardistance = distance
                    farthestList = [pod,pod2,fardistance]

            #adds the pod, and then the corresponding distances
            adjacencyList.append([pod['podId'],listOfDistFromPods])

        #returns [pod1,pod2,distance]
        return [adjacencyList,farthestList]

    def closestpod(self,podid,adjacencyList, listOfPods):
        closest = ['placeholder',sys.float_info.max]

        #finds the pod with matching id
        for p in adjacencyList:
            if(p[0] == podid):
                #finds the closest pod to the current pod
                for distance in p[1]:
                    if(distance[1] < closest[1] and distance[1] != 0 and distance[0] in listOfPods):
                        closest = distance

        #returns closest podid and distance
        return closest
