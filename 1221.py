from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import csv
import math
import pandas as pd
from scipy.spatial import distance_matrix


speed = 50
max_dist = 3000  #maximum_distance
time = max_dist/speed

