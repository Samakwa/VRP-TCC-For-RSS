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

    v = 24
    #v= int(nodelist.values())
    for k in nodelist.items():

        v+= v
        if v < allowedtime:
            cluster1.append([k,v])
            nodes1.append (k)


        else:
            cluster2.append([k, v])
            nodes2.append(k)

        print("Cum weight2: ", v)


    #def time_cum:
       # for k, v in cluster1:
          #  v += v
           # if v < allowedtime:
            #    print(v)
             #   print(k, "----->")

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
    nodes3 = []
    nodes4 = []
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

    """

"""
# Mayavi

    graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([graph_pos[v] for v in sorted(G)])
maxCapacity = 96000
newcluster =[]
"""
def check_cap(graph, node):
    cum_cap = int(capacity(graph))


    for item in node:
            if Cum_cap < maxCapacity:
                pass
                Cum_cap += Cum_cap
            else:
                newcluster.append(item)
                Cum_cap += Cum_cap
        print("PODs within route1: ---")
        for item in route1:
            print(item)

        print("PODs within route2: ---")
        for item in route2:
            print(item)

        for item in secondcluster:
            if Cum_cap < maxCapacity:
                route1.append(item)
                Cum_cap += Cum_cap
            else:
                route3.append(item)
                Cum_cap += Cum_cap
"""
def capacity(graph):
    """
    with open('PODlist2.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.read(1024))
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            popn.append(row[2])
    """
    # randomly add population as capacity to nodes
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.randint(6000, 7000)
    """
    for node in G.nodes():
        edges = G.in_edges(node, data=True)
       
        if len(edges) > 0:  # some nodes have zero edges going into it
            min_weight = min([edge[2]['weight'] for edge in edges])
            for edge in edges:
                if edge[2]['weight'] > min_weight:
                    G.remove_edge(edge[0], edge[1])
                    c = Counter(g.edges())  # Contains frequencies of each directed edge.

                    for u, v, d in g.edges(data=True):
                        d['weight'] = c[u, v]
    #esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    """
    x =0
    while x < (len(graph.nodes())):
        cap = [(d) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0]

        x +=1

        return cap

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
child2()
#draw_graph3d(G)

#distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2,  start)
#distance2()
#distance3()
#matrix()
#capacity(G)
