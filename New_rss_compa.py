import math
import csv
import webbrowser

allowedtime= 12 #24  #48
allowed_popn = 11000
route_count = 10
cluster = []
#cluster1 =[]
#cluster2 =[]
nodes1=[]
nodes2 =[]
# RSS = input ( "Enter Coordinates of RSS; lat, long: ")
RSS = (29.779630, -95.436960)

with open('Route_Distances.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.

    for row in OurPOD:
        source_ID = row[0]
        source_name = row[1]
        Source_Popn = row[2]
        destin_ID = row[3]
        Destin_Name = row[4]
        Distn = row[5]
        # x = float(x)
        # y = float(y)
        # destination = [x, y]

dataset = dict()

with open('Route_Distances.csv', 'r+') as in_file:
    OurPOD = csv.reader(in_file)
    has_header = csv.Sniffer().has_header(in_file.readline())
    in_file.seek(0)  # Rewind.
    if has_header:
        next(OurPOD)  # Skip header row.
    for row in OurPOD:
        raw_data = dataset.get(row[1], None)

        if raw_data is None:
            dataset[1] = {row[4]: row[5]}
        else:
            raw_data[row[4]] = row[5]
            dataset[row[1]] = raw_data

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def main():
    d = AutoVivification()
    filename = 'Route_Distances.csv'
    with open(filename, 'r+') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)        # skip the header
        for row in reader:
            d[row[1]][row[4]] = row[5]

    print(d)


def distance(): #(RSS, destination):
    # readdata()
    cum_dist = 0
    with open('Route_Distances.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        source = []
        for row in OurPOD:
            source_ID = row[0]
            source_name = row[1]
            Source_Popn = row[2]
            destin_ID = row[3]
            Destin_Name = row[4]
            Distn = row[5]
            source.append(row[2])
            if row[0] == '152':
                print (source)


            #d = radius * c
            #print("Distance from RSS:", d)
            # return d

            # Change the starting node after each iteration

            dist = int(row[5])
            popn = int (row[2])

            cum_popn =0
            cum_dist = cum_dist + dist
            cum_popn+= popn
            # Assume average speed = 55 miles/hr
            speed = 55

            print("Cumulative route distance through nearest neighbor:", cum_dist)
            time = cum_dist / speed

            listDict = {}
            for i in range(1,route_count):
                cluster[i] = []
                if time < allowedtime:
                    listDict["Route_" + str(i)] = []
                    cluster[i].append([source_name])
                    #nodes1.append(addr)

                else:
                    listDict["Route_" + str(i+1)] = []

                    cluster[i+1].append(source_name)
                i+=1

                listDict2 = {}
                for i in range(1, route_count):
                    cluster[i] = []
                    if cum_popn< allowed_popn:
                        listDict["Route_" + str(i)] = []
                        cluster[i].append([source_name])
                        # nodes1.append(addr)

                    else:
                        listDict["Route_" + str(i + 1)] = []

                        cluster[i + 1].append(source_name)
                    i += 1
                print (i)
    print("PODs in cluster1:")
    for item in cluster[i]:
        print(item)





def openmap():
    with open('newroutes.csv', 'w') as out_file:
        new_list = csv.writer(out_file)

    webbrowser.open("https://planner.myrouteonline.com/route-planner")
    webbrowser.open("https://www.google.es/maps/dir/'-95.436960,29.779630'/'-95.063668,29.900089")


distance()
"""
def game_compare(s1,s2):
    if source[i] not in cluster[j]:
        return ('this is a tie')
    elif s1 =='stone':
        if s2 == 'scissors':
            return ("Stone wins")
        else:
            return("Spread wins")

    elif s1 == "scissors":
        if s2 == "spread":
            return("Scissors wins!")
        else:
            return ("Stone Wins!")

    elif s1 == "Spread":
        if s2 == 'stone':
            return ('Spread wins')
        else:
            return ("Scissors wins")

distance(RSS, destination=readdata())

"""