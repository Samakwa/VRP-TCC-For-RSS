import csv
import math
import pandas as pd

loc1 = []

with open('test1_popn.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.


    #loc1= []
    data = {}
    for row in OurPOD:
        loc1.append([(row[3]), row[4]])

        locations = [(row[3]), row[4]]

    print (locations)
print (loc1)