from __future__ import division
import networkx as nx
from networkx.algorithms import approximation
# import plotly.plotly as py
# import plotly.graph_objs as go
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
print(len(G))
print(G.size())
print(list(G.nodes))

# G.add_nodes_from ('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# for item in G:
# print(item)

e = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
     'X', 'Y', 'Z']

# for route in G.edges():
#   print (route)

# randomly add weight to edges
for (u, v, w) in G.edges(data=True):
    w['weight'] = random.randint(3, 12)

# calculate distances to get two farthest nodes

# Calculate eccentricity: Max distance from the reference node to all other nodes
ecce = nx.eccentricity(G, G.nodes())
print('ecce', ecce)
di = nx.diameter(G)
print('diameter', di, ecce)

mat = nx.floyd_warshall_numpy(G, nodelist=G.nodes(), weight='weight')
print('matrix', mat)

len01 = nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')
for item in len01:
    print(item)

nx.draw(G, with_labels=True)
pos = nx.spring_layout(G)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
plt.savefig("graph.png")
plt.show()

print("Number of nodes in main graph: ", G.number_of_nodes())


def confirm_fnodes():
    for node in e:
        start2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
        l = nx.single_source_dijkstra_path_length(G, start)
        lmin = min(l, key=l.get)
        rmax = max(l, key=l.get)
        max_value = max(l.values())
        for k, v in l.items():
            if k == rmax and v < 72:
                # print(v)
                if v < 12:
                    break
                break
            break


allowedtime = 48
cluster1 = []
cluster2 = []
nodes1 = []
nodes2 = []


def distance_main(graph, cluster1, cluster2, nodes1, nodes2):
    # start = random.choice('ABCDEFGHIJKLMNOPQRSTUVWSYZ')
    start = input("Enter starting node for most distant nodes: ")
    nodelist = nx.single_source_dijkstra_path_length(graph, start)

    print("Two nodes farthest apart are: ")
    lmin = min(nodelist, key=nodelist.get)
    rmax = max(nodelist, key=nodelist.get)
    print('Node', rmax, "------>", "Node", lmin)


    # calculate distance 1
    dist1 = nx.single_source_dijkstra_path_length(graph, lmin)

    # calculate distance 2
    dist2 = nx.single_source_dijkstra_path_length(graph, rmax)

    cum_dist = 0
    for k, v in nodelist.items():
        if dist1 > dist2:

        # v= int(nodelist.values())

            cum_dist = cum_dist + v
            print(cum_dist)
            if cum_dist < allowedtime:
                cluster1.append([k, v])
                nodes1.append(k)


            else:
                cluster2.append([k, v])
                nodes2.append(k)
        else:
            cum_dist = cum_dist + v
            print(cum_dist)
            if cum_dist < allowedtime:
                cluster1.append([k, v])
                nodes1.append(k)


            else:
                cluster2.append([k, v])
                nodes2.append(k)
    """  
    for k in cluster1:
        cluster1[k].append(-1)
        cluster1[k].append(-2)
    cluster1[-1] = list(range(1, graph.number_of_nodes()))
    cluster1[-2] = list(range(1, graph.number_of_nodes()))

    def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not start in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    for path in find_all_paths(cluster1, -1, -2):
        if len(path) == len(cluster1):
            print(path[1:-1])

        ham_paths = [path[1:-1] for path in find_all_paths(cluster1, -1, -2)
                     if len(path) == len(cluster1)]
        ham_path = random.choice(ham_paths)

        for k, vs in cluster1.items():
            for v in vs:
                if v in [-1, -2] or k in [-1, -2]:
                    continue
                if abs(ham_path.index(k) - ham_path.index(v)) == 1:
                    graph.add_edge(k, v, color='red', width=1.5)
                else:
                    graph.add_edge(k, v, color='black', width=0.5)

        pos = nx.circular_layout(graph)
        edges = graph.edges()
        colors = [graph[u][v]['color'] for u, v in edges]
        widths = [graph[u][v]['width'] for u, v in edges]
        nx.draw(graph, pos, edges=edges, edge_color=colors, width=widths)
        
        """
        #with open("cluster1.csv", "w", newline="") as f:
            #writer = csv.writer(f)
            #writer.writerows(cluster1)
"""

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

            # print(x)
            x = float(x)
            y = float(y)
            x0 = -118.453
            y0 = 34.21

            distance2 += math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            distance = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return distance

"""
def child1():
    distance_main(G, cluster1, cluster2, nodes1, nodes2)
    a = nodes1
    b = nodes2

    n2 = ''.join(a)
    k2 = ''.join(b)
    G1 = nx.complete_graph(n2)
    G2 = nx.complete_graph(k2)

    print("Number of nodes in first cluster: ")  # G2.size())
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
    # start = random.choice(k2)

    # randomly add weight to edges
    for (u, v, w) in G2.edges(data=True):
        w['weight'] = random.randint(3, 12)
    len3 = nx.all_pairs_dijkstra_path_length(G2, cutoff=None, weight='weight')
    # print (len)
    print(" Third routing")
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

    print("Number of nodes in third cluster: ")  # G2.size())
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

    graph_pos = nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([graph_pos[v] for v in sorted(G)])


maxCapacity = 96000
newcluster = []

popn = []


def capacity(graph):
    with open('Newcluster1.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.read(1024))
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            popn.append(row[2])

            x = 0
            while x < (len(popn)):
                cap = popn[x]
                x += 1
                # print (cap)
                return cap


route1 = []
route2 = []


def cap2(graph):
    Cum_cap = int(capacity(graph))
    maxCapacity = int(input("Enter maximum capacity of a truck:  "))
    with open('Newcluster1.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.read(1024))
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            popn.append(row[2])

        # i=0
        # i+=i
        # capacity = popn[i]
        for item in cluster1:
            if Cum_cap < maxCapacity:
                route1.append(item)
                Cum_cap += Cum_cap
            else:
                route2.append(item)
                Cum_cap += Cum_cap
        print("PODs within route1: ---")
        for item in route1:
            print(item)

        print("PODs within route2: ---")
        for item in route2:
            print(item)
        


        print("PODs within prunned route, not yet assigned: ---")

        """
    #with open('newroutes.csv', 'w') as out_file:
   #     new_list = csv.writer(out_file)

    #    webbrowser.open("https://planner.myrouteonline.com/route-planner")


"""
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

    if n == 1: return {nodes[0]: {}}

    nodes1 = nodes[0:n / 2]
    nodes2 = nodes[n / 2:]
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
# ax = plt.gca()
# ax.set_axis_off()
# plt.show()

# farthest_nodes()
# distance3()
# child1()
child2()
# draw_graph3d(G)

# distance_main(G, cluster1, cluster2, clusternodes1, clusternodes2,  start)
# distance2()
# distance3()
# matrix()
# capacity(G)
