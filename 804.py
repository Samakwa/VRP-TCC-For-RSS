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

#G.add_nodes_from ('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#for item in G:
   # print(item)

e= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z']

#for route in G.edges():
#   print (route)

# randomly add weight to edges
for (u,v,w) in G.edges(data=True):
    w['weight'] = random.randint(24,72)

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


    #val = int(nodelist.values())
    for k, v in nodelist.items():
        if v < allowedtime:
            cluster1.append([k,v])
            nodes1.append (k)
        else:
            cluster2.append([k, v])
            nodes2.append(k)


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


def child2():
    child1()
    b = nodes2

    k2 = ''.join(b)
    G2 = nx.complete_graph(k2)
    #start = random.choice(k2)


    # randomly add weight to edges
    for (u, v, w) in G2.edges(data=True):
        w['weight'] = random.randint(24, 72)
    len3 = nx.all_pairs_dijkstra_path_length(G2, cutoff=None, weight='weight')
    # print (len)
    print (" Third routing")
    for item in len3:
        print(item)
    cluster3 = []
    cluster4 = []
    clusternodes3 = []
    clusternodes4 = []
    distance_main(G2, cluster3, cluster4, clusternodes3, clusternodes4)


    c = clusternodes3
    d = clusternodes4
    n2 = ''.join(c)
    k2 = ''.join(d)
    G3 = nx.complete_graph(n2)
    G4 = nx.complete_graph(k2)

    print('after further division, according to time constraint, we have: ')

    print("Number of nodes in third cluster: " ) #G2.size())
    print(G3.number_of_nodes())
    print("Cluster nodes :")
    print(clusternodes3)
    print(len(G3))


    print("Number of nodes in fourth cluster: ")
    print(G4.number_of_nodes())
    print("Cluster nodes :")
    print(clusternodes4)
    print(len(G4))



    nx.draw(G3, with_labels=True)
    plt.savefig("graph_Child3.png")
    plt.show()
    nx.draw(G4, with_labels=True)
    plt.savefig("graph_Child4.png")
    plt.show()

    """

"""
# Mayavi

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])

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



def capacity(graph):
    d = list(l)
    d['A'].append(1000)

    #d['a'].append(5)

    {'a': [1, 5], 'b': [2]}




#ax = plt.gca()
#ax.set_axis_off()
#plt.show()

#farthest_nodes()
#distance3()
#child1()
child2()
#draw_graph3d(G)

#distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2,  start)
#distance2()
#distance3()
#matrix()
