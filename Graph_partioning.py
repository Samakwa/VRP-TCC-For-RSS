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
from scipy.spatial import distance
import webbrowser


G = nx.DiGraph()

G.add_nodes_from ('ABCDEFGHIJKLMNOPQRSTUVWSYZ') #(range(0,50))
print(list(G.nodes))


#G.add_edges_from([(1, 2), (2, 1), (2, 3)])



G.add_edges_from([('A', 'B'),('C','D'),('G','D')], weight=1)
G.add_edges_from([('D','A'),('D','E'),('B','D'),('D','E')], weight=2)
G.add_edges_from([('B','C'),('E','F'), ('D', 'H'), ('H', 'J'), ( 'K', 'T')], weight=3)
G.add_edges_from([('C','F'), ('G', 'S'), ('L', 'R')], weight=4)
G.add_edges_from([('T', 'Y'),('T','U'),('P','O')], weight=5)
G.add_edges_from([('W','R'),('W','Y'),('B','F'),('D','P')], weight=5)
G.add_edges_from([('K','C'),('N','F'), ('V', 'M'),('V','G'), ('H','A')], weight=6)
G.add_edges_from([('A','F'), ('U','R'), ('C', 'Q')], weight=7)
print (list(G.edges))

#l =nx.dag_longest_path_length(G)
#print(l)

l= nx.single_source_dijkstra_path_length(G, 'A')
print(l)
nx.draw(G,with_labels=True)
plt.savefig("graph.png")
plt.show()



"""


ax = plt.gca()
ax.set_axis_off()
plt.show()
"""