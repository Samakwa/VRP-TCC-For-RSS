from __future__ import division
import networkx as nx
from networkx.algorithms import approximation
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import csv
import pandas as pd
import math
from math import sqrt
import sys
import scipy
from scipy.spatial import distance
import webbrowser


G = nx.DiGraph()

G.add_nodes_from ('ABCDEFGHIJKLMNOPQRSTUVWSYZ') #(range(0,50))
#print(list(G.nodes))


G.add_edges_from([(1, 2), (2, 1), (2, 3)])
x


G.add_edges_from([('A', 'B'),('C','D'),('G','D')], weight=1)
G.add_edges_from([('D','A'),('D','E'),('B','D'),('D','E')], weight=2)
G.add_edges_from([('B','C'),('E','F'), ('D', 'H'), ('H', 'J'), ( 'K', 'T')], weight=3)
G.add_edges_from([('C','F'), ('G', 'S'), ('L', 'R')], weight=4)
G.add_edges_from([('T', 'Y'),('T','U'),('P','O')], weight=5)
G.add_edges_from([('W','R'),('W','Y'),('B','F'),('D','P')], weight=5)
G.add_edges_from([('K','C'),('N','F'), ('V', 'M'),('V','G'), ('H','A')], weight=6)
G.add_edges_from([('A','F'), ('U','R'), ('C', 'Q'), ('M', 'R')], weight=7)

# calculate distances to get two farthest nodes

len= nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')
print (len)
for item in len:
    print (item)


start = input ("Enter starting node for most distant nodes: ")

l= nx.single_source_dijkstra_path_length(G, start)
print(l)
len = dict(l)
print (max(len, key=len.get))


print ("Two nodes farthest apart are: ")
lmin = min(l, key=l.get)
rmax = max(l, key=l.get)
print( 'Node', lmin, "------>", "Node", rmax)

nx.draw(G,with_labels=True)
plt.savefig("graph.png")
plt.show()

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

alloweddist= 10
cluster1 =[]
cluster2 =[]
def distance3():
    nodelist = nx.single_source_dijkstra_path_length(G, 'V')
    #val = int(nodelist.values())
    for k, v in nodelist.items():
        if v < alloweddist:
            cluster1.append([k,v])
        else:
            cluster2.append([k, v])

    print(cluster1)
    print(cluster2)

def matrix():
    mat= nx.floyd_warshall_numpy(G, nodelist= G.nodes()) #weight=G.edges())
    print (mat)



def capacity():
    d = list(l)
    d['A'].append(1000)

    #d['a'].append(5)

    {'a': [1, 5], 'b': [2]}
"""



ax = plt.gca()
ax.set_axis_off()
plt.show()
"""

distance2()
distance3()
matrix()