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
    w['weight'] = random.randint(5,12)

# calculate distances to get two farthest nodes

len= nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')
#print (len)
for item in len:
    print (item)

start = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
#start = input ("Enter starting node for most distant nodes: ")

l= nx.single_source_dijkstra_path_length(G, start)

print("Two nodes farthest apart are: ")
lmin = min(l, key=l.get)
rmax = max(l, key=l.get)
print('Node', lmin, "------>", "Node", rmax)


def confirm_fnodes():
    for node in e:
        start2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
        l= nx.single_source_dijkstra_path_length(G, start)
        lmin = min(l, key=l.get)
        rmax = max(l, key=l.get)
        max_value = max(l.values())
        for k, v in l.items():
            if k == rmax:
                #print(v)
                if v < 12:
                    break
                break
            break

def distance2():
    n =26
    ncols = 325
    pos = {i: (i , (34) % ncols) for i in G.nodes()}
    #Compute the node-to-node distances
    lengths={}
    for edge in G.edges():
        startnode=edge[0:]
        endnode=edge[:]
        lengths[edge]=round(math.sqrt(((pos[endnode][0:]-pos[startnode][:])**2)+
                                      ((pos[endnode][:]-pos[startnode][0:])**2)),2) #The distance
    items = sorted(lengths.items())
    values = lengths.values()
    df = pd.DataFrame({'Lengths': values})

    df['Lengths'].hist(df, bins=10)
    for item in lengths:
        print(item)

alloweddist= 10
cluster1 =[]
cluster2 =[]
cluster1_nodes=[]
cluster2_nodes =[]
def distance3():

    nodelist = nx.single_source_dijkstra_path_length(G, lmin)
    #val = int(nodelist.values())
    for k, v in nodelist.items():
        if v < alloweddist:
            cluster1.append([k,v])
            cluster1_nodes.append (k)
        else:
            cluster2.append([k, v])
            cluster2_nodes.append(k)

    print(cluster1)
    print(cluster2)
    print("Cluster 2 nodes:")
    print(cluster2_nodes)


def matrix():
    mat= nx.floyd_warshall_numpy(G, nodelist= G.nodes()) #weight=G.edges())
    print (mat)

nx.draw(G,with_labels=True)
plt.savefig("graph.png")
plt.show()

print (G.number_of_nodes())
print (G.number_of_edges())



def child1():
    distance3()
    a=cluster1_nodes
    b = cluster2_nodes

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

    # Calculate eccentricity: Max distance from the reference node to all other nodes
    ecce = nx.eccentricity(G3, rmax)
    print(ecce)
    G3_dist = nx.single_source_shortest_path_length(G3, rmax)
    print(G3_dist)
    Dist_allG3 = nx.all_pairs_shortest_path_length(G3, alloweddist)
    for item in Dist_allG3:
        print(item)

"""
# Mayavi
def draw_graph3d(graph):

    H=nx.Graph()
    G = nx.convert_node_labels_to_integers(H)

    # add edges
    for node, edges in H.items():
        for edge, val in edges.items():
            if val == 1:
                H.add_edge(node, edge)

    #G=nx.convert_node_labels_to_integers(H)

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
"""
"""

def distance2():
    n =36
    ncols = 6
    pos = {i: (i , (34) % ncols) for i in G.nodes()}
    #Compute the node-to-node distances
    lengths={}
    for edge in G.edges():
        startnode=edge[0:]
        endnode=edge[:]
        lengths[edge]=round(math.sqrt(((pos[endnode][0:]-pos[startnode][:])**2)+
                                      ((pos[endnode][:]-pos[startnode][0:])**2)),2) #The distance
    items = sorted(lengths.items())
    values = lengths.values()
    df = pd.DataFrame({'Lengths': values})

    df['Lengths'].hist(df, bins=10)
    for item in lengths:
        print(item)




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
#draw_graph3d(G)
"""

#distance2()
#distance3()
matrix()
"""