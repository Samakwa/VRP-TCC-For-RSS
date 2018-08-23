from __future__ import division
import networkx as nx
from networkx.algorithms import approximation
#import plotly.plotly as py
#import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import date
import csv
from collections import Counter
import pandas as pd
import math
from math import sqrt
import sys
import scipy
from scipy.spatial import distance
import webbrowser
import itertools

n = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
G = nx.complete_graph(n)
print (len(G))
print(G.size())
print(list(G.nodes))


e= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z']

#for route in G.edges():
#   print (route)

# randomly add weight to edges
for (u,v,w) in G.edges(data=True):
    w['weight'] = random.randint(2,17)


# calculate distances to get two farthest nodes

# Calculate eccentricity: Max distance from the reference node to all other nodes
ecce = nx.eccentricity(G, G.nodes())
print('ecce',ecce)
di = nx.diameter(G)
print ('diameter', di, ecce)


mat = nx.floyd_warshall_numpy(G, nodelist=G.nodes(), weight='weight')
print('matrix', mat)


len01= nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')
for item in len01:
    print (item)


nx.draw(G,with_labels=True)
pos=nx.spring_layout(G)
#labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
plt.savefig("graph.png")
plt.show()

print ("Number of nodes in main graph: ", G.number_of_nodes())


def confirm_fnodes():
    for node in e:
        start2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
        l= nx.single_source_dijkstra_path_length(G, start)
        lmin = min(l, key=l.get)
        rmax = max(l, key=l.get)
        max_value = max(l.values())
        for k, v in l.items():
            if k == rmax and v < 72:
                #print(v)
                if v < 12:
                    break
                break
            break



allowedtime= 48
cluster1 =[]
cluster2 =[]
nodes1=[]
nodes2 =[]
def distance_main(graph, cluster1, cluster2, nodes1, nodes2):
    # start = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
    start = input("Enter starting node for most distant nodes: ")
    nodelist = nx.single_source_dijkstra_path_length(graph, start)

    print("Two nodes farthest apart are: ")
    lmin = min(nodelist, key=nodelist.get)
    rmax = max(nodelist, key=nodelist.get)
    print('Node', rmax, "------>", "Node", lmin)

    cum_dist = 0
    #v= int(nodelist.values())
    for k, v in nodelist.items():
        cum_dist = cum_dist + v
        print(cum_dist)
        if cum_dist < allowedtime:
            cluster1.append([k,v])
            nodes1.append (k)


        else:
            cluster2.append([k, v])
            nodes2.append(k)

    with open("cluster1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cluster1)
route1 =[]
route2 =[]
route3 =[]
route4 =[]
def dist_cumu():
    with open('PODlist2.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)

        distance = 0.0
        distance2 = 0.0
        has_header = csv.Sniffer().has_header(in_file.read(1024))
        in_file.seek(0)  # Rewind.

        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            x = row[3]
            y = row[4]

            #print(x)
            x =float(x)
            y = float(y)
            x0 = -118.453
            y0 = 34.21

            distance2 += math.sqrt((x - x0)**2 + (y - y0)**2)
            distance = math.sqrt((x - x0)**2 + (y - y0)**2)
    return distance
def child1():
    distance_main(G, cluster1, cluster2, nodes1, nodes2)
    a= nodes1
    b = nodes2

    n2 = ''.join(a)
    k2 = ''.join(b)
    G1 = nx.complete_graph(n2)
    G2 = nx.complete_graph(k2)


    print("Number of nodes in first cluster: " ) #G2.size())
    print(G1.number_of_nodes())
    print(len(G1))

    print("Number of nodes in second  cluster: ")
    print(G2.number_of_edges())

    print(len(G2))

    nx.draw(G1, with_labels=True)
    plt.savefig("graph_Child1.png")
    plt.show()
    nx.draw(G2, with_labels=True)
    plt.savefig("graph_Child2.png")
    plt.show()


cluster3 = []
cluster4 = []
nodes3 = []
nodes4 =[]
def child2():
    child1()
    b = nodes2

    k2 = ''.join(b)
    G2 = nx.complete_graph(k2)
    #start = random.choice(k2)


    # randomly add weight to edges
    for (u, v, w) in G2.edges(data=True):
        w['weight'] = random.randint(3, 15)
    len3 = nx.all_pairs_dijkstra_path_length(G2, cutoff=None, weight='weight')
    # print (len)
    print (" Third routing")
    for item in len3:
        print(item)

    distance_main(G2, cluster3, cluster4, nodes3, nodes4)


    c = nodes3
    d = nodes4
    n2 = ''.join(c)
    k2 = ''.join(d)
    G3 = nx.complete_graph(n2)
    G4 = nx.complete_graph(k2)

    print('after further division, according to time constraint, we have: ')

    print("Number of nodes in third cluster: " ) #G2.size())
    print(G3.number_of_nodes())
    print("Cluster nodes :")
    print(nodes3)
    print(len(G3))


    print("Number of nodes in fourth cluster: ")
    print(G4.number_of_nodes())
    print("Cluster nodes :")
    print(nodes4)
    print(len(G4))



    nx.draw(G3, with_labels=True)
    plt.savefig("graph_Child3.png")
    plt.show()
    nx.draw(G4, with_labels=True)
    plt.savefig("graph_Child4.png")
    plt.show()


    #cap2(G4)
    """ewrfvfv

"""
# Mayavi

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])
maxCapacity = 96000
newcluster =[]


popn = []


def capacity(graph, firstcluster, secondcluster, firstroute, secondroute):
    child2()
    # randomly add weight to edges
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.randint(6000, 12000)

    print ('analyzing capacity constraint: ')
    g= graph.get_edge_data

    popn = random.randint(6000, 12000)

    Cum_cap = popn


    for item in firstcluster:

        if Cum_cap < maxCapacity:
            firstroute.append(item)
            Cum_cap += Cum_cap
        else:
            # route2.append(item)
            secondcluster.append(item)
            Cum_cap += Cum_cap
    print("PODs within firstroute: ---")
    for item in firstroute:
        print(item)
        """
        print("PODs within route2: ---")
        for item in cluster2:
                print(item)
        """
    for item in secondcluster:
        if Cum_cap < maxCapacity:
            secondroute.append(item)
            Cum_cap += Cum_cap
        else:
            cluster3.append(item)
            Cum_cap += Cum_cap
    print("PODs within next route: ---")
    for item in secondroute:
        print(item)

def cap():
    a = nodes1
    b = nodes2
    c = nodes3
    d = nodes4
    n2 = ''.join(a)
    k2 = ''.join(b)
    G1 = nx.complete_graph(n2)
    G2 = nx.complete_graph(k2)
    n2 = ''.join(c)
    k2 = ''.join(d)

    capacity(G, cluster1, cluster2, route1, route2)
    resp = input (' Run another capacity analysis? : ')
    while resp =="Y" or resp == "y":

        capacity(G1, cluster1, cluster2, route1, route2)
        capacity(G2, cluster2, cluster3, route2, route3)
        capacity(G3, cluster3, cluster4, route3, route4)

    """

"""
    """"


    with open('newroutes.csv', 'w') as out_file:
        new_list = csv.writer(out_file)

   #webbrowser.open("https://planner.myrouteonline.com/route-planner")



    
    # randomly add population as capacity to nodes
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.randint(6000, 7000)

    for node in G.nodes():
        edges = G.in_edges(node, data=True)
       
        if len(edges) > 0:  # some nodes have zero edges going into it
            min_weight = min([edge[2]['weight'] for edge in edges])
            for edge in edges:
                if edge[2]['weight'] > min_weight:
                    G.remove_edge(edge[0], edge[1])
                    c = Counter(g.edges())  # Contains frequencies of each directed edge.

"""
def make_graph(nodes):

    def make_link(graph, i1, i2):
        graph[i1][i2] = 1
        graph[i2][i1] = 1

    n = len(nodes)

    if n == 1: return {nodes[0]:{}}

    nodes1 = nodes[0:n/2]
    nodes2 = nodes[n/2:]
    G1 = make_graph(nodes1)
    G2 = make_graph(nodes2)

    # merge G1 and G2 into a single graph
    G = dict(G1.items() + G2.items())

    # link G1 and G2
    random.shuffle(nodes1)
    random.shuffle(nodes2)
    for i in range(len(nodes1)):
        make_link(G, nodes1[i], nodes2[i])

    return G


#ax = plt.gca()
#ax.set_axis_off()
#plt.show()

#farthest_nodes()
#distance3()
#child1()
#child2()
#draw_graph3d(G)

#distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2,  start)
#distance2()
#distance3()
#matrix()
#capacity(G, cluster1, cluster2, route1, route2)
#cap2(G, cluster1, cluster2, route1, route2)
cap()
with open('newroutes.csv', 'w') as out_file:
    new_list = csv.writer(out_file)

webbrowser.open("https://planner.myrouteonline.com/route-planner")