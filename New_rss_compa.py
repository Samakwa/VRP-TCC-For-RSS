import math
import csv
import webbrowser

allowedtime = 12  # 24  #48
cluster1 = []
cluster2 = []
nodes1 = []
nodes2 = []

# RSS = input ( "Enter Cordinates of RSS; lat, long: ")
RSS = (29.779630, -95.436960)


def readdata():
    with open('Route_Distances.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            x = row[3]
            y = row[2]

            # print(x)
            # x = float(x)
            # y = float(y)
            destination = [x, y]


def distance(RSS, destination):
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
            x = row[3]
            source.append(row[2])

            y = row[2]
            addr = row[1]
            # print(x)
            x = float(x)
            y = float(y)
            destination = [x, y]
            lat1, lon1 = RSS
            lat2, lon2 = destination
            RSS = (29.779630, -95.436960)
            radius = 3959  # Miles

            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
                * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            d = radius * c
            print("Distance from RSS:", d)
            # return d

            # Change the starting node after each iteration

            RSS = (x, y)
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
                * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = radius * c

            cum_dist = cum_dist + dist
            # Assume average speed = 55 miles/hr
            speed = 55

            print("Cumulative route distance through nearest neighbor:", cum_dist)
            time = cum_dist / speed
            if time < allowedtime:
                cluster1.append([x, y])
                nodes1.append(addr)


            else:
                i = 2

                cluster2.append([x, y])
                nodes2.append(addr)
    print("PODs in cluster1:")
    for item in cluster1:
        print(item)
    for addr in nodes1:
        print(addr)
    print("PODs in cluster2:")
    for item in cluster2:
        print(item)
    for addr in nodes2:
        print(addr)


def openmap():
    with open('newroutes.csv', 'w') as out_file:
        new_list = csv.writer(out_file)

    webbrowser.open("https://planner.myrouteonline.com/route-planner")
    webbrowser.open("https://www.google.es/maps/dir/'-95.436960,29.779630'/'-95.063668,29.900089")


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