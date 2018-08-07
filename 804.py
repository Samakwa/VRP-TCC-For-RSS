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

len01= nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')
for item in len01:
    p
for item in G.nodes():
    l = nx.single_source_dijkstra_path_length(G, item)
    #print(item)
    #len001 = dict(len)
    lmin = min(l, key=l.get)
    rmax = max(l, key=l.get)
    print (lmin, rmax)

    start = rmax
    l = nx.single_source_dijkstra_path_length(G, start)

    print("Two nodes farthest apart are: ")
    lmin = min(l, key=l.get)
    rmax = max(l, key=l.get)
    print('Node', lmin, "------>", "Node", rmax)

# Calculate eccentricity: Max distance from the reference node to all other nodes
ecce = nx.eccentricity(G, G.nodes())
print('ecce',ecce)
di = nx.diameter(G)
print ('diameter', di, ecce)


mat = nx.floyd_warshall_numpy(G, nodelist=G.nodes(), weight='weight')
print('matrix', mat)



#lmin = min(len2, key=len2.get)
#rmax = max(len2, key=len2.get)
#print (lmin, rmax)
dict1 ={}


#start = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
#start = input ("Enter starting node for most distant nodes: ")



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
clusternodes1=[]
clusternodes2 =[]
def distance_main(graph, cluster1, cluster2, clusternodes1, clusternode2,  start):

    nodelist = nx.single_source_dijkstra_path_length(graph, start)
    #val = int(nodelist.values())
    for k, v in nodelist.items():
        if v < allowedtime:
            cluster1.append([k,v])
            clusternodes1.append (k)
        else:
            cluster2.append([k, v])
            clusternode2.append(k)

    print(cluster1)
    print(cluster2)
    print("Cluster 2 nodes :")
    print(clusternodes2)



nx.draw(G,with_labels=True)
plt.savefig("graph.png")
plt.show()

print (G.number_of_nodes())
print (G.number_of_edges())



def child1():
    distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2, start)
    a=clusternodes1
    b = clusternodes2

    n2 = ''.join(a)
    k2 = ''.join(b)
    G2 = nx.complete_graph(n2)
    G3 = nx.complete_graph(k2)


    print("size of first cluster", G2.size())
    print(list(G2.nodes))
    print("size of second cluster", G3.size())
    print(list(G3.nodes))
    nx.draw(G2, with_labels=True)
    plt.savefig("graph_Child1.png")
    plt.show()
    nx.draw(G3, with_labels=True)
    plt.savefig("graph_Child2.png")
    plt.show()




""""
def child2():
    child1()
    b = clusternodes2

    k2 = ''.join(b)
    G3 = nx.complete_graph(k2)
    start = random.choice(k2)
    l3 = nx.single_source_dijkstra_path_length(G3, start)

    # randomly add weight to edges
    for (u, v, w) in G3.edges(data=True):
        w['weight'] = random.randint(24, 72)
    len3 = nx.all_pairs_dijkstra_path_length(G3, cutoff=None, weight='weight')
    # print (len)
    print (" Third routing")
    for item in len3:
        print(item)
    
    print("Two nodes farthest apart are: ")
    lmin = min(l, key=l.get)
    rmax = max(l, key=l.get)

    nodelist = nx.single_source_dijkstra_path_length(G3, lmin)
    # val = int(nodelist.values())
    for k, v in nodelist.items():
        if v < allowedtime:
            cluster13.append([k, v])
            cluster13_nodes.append(k)
        else:
            cluster4.append([k, v])
            cluster4_nodes.append(k)

    print(cluster1)
    print(cluster2)
    print("Cluster 2 nodes:")
    print(cluster2_nodes)

    print('Node', lmin, "------>", "Node", rmax)

    print("nodes in this graph",
          list(G3.nodes))
    print("size of second cluster", G3.size())
    print(list(G3.nodes))
    nx.draw(G2, with_labels=True)
    plt.savefig("graph_Child1.png")
    plt.show()
    nx.draw(G3, with_labels=True)
    plt.savefig("graph_Child2.png")
    plt.show()
    
    """

"""
# Mayavi

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])
    print(G.edges())
    print(list(G.nodes))

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



def capacity():
    d = list(l)
    d['A'].append(1000)

    #d['a'].append(5)

    {'a': [1, 5], 'b': [2]}

"""


ax = plt.gca()
ax.set_axis_off()
plt.show()

#farthest_nodes()
#distance3()
child1()
#child2()
#draw_graph3d(G)

#distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2,  start)
#distance2()
#distance3()
#matrix()
