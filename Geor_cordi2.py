import math
import csv
import webbrowser

allowedtime= 12 #24  #48
allowed_popn= 11000
route_count = 10
#cluster = []
cluster1 =[]
cluster2 =[]
nodes1=[]
nodes2 =[]
new_cluster = 1


#RSS = input ( "Enter Coordinates of RSS; lat, long: ")
RSS = (29.779630, -95.436960)
def readdata():
    with open('test1_popn.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            x = row[3]
            y = row[2]

            destination = [x, y]


def distance(RSS, destination):
    #readdata()
    new_cluster = 1

    cum_dist = 0
    with open('Route_Distances2.csv', 'r+') as in_file:
    #with open('test1_popn.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            x = row[3]
            y = row[2]
            addr = row[1]
            # print(x)
            x = float(x)
            y = float(y)
            destination = [x,y]
            lat1, lon1 = RSS
            lat2, lon2 = destination
            RSS = (29.779630, -95.436960)
            radius = 3959 # Miles

            dlat = math.radians(lat2-lat1)
            dlon = math.radians(lon2-lon1)
            a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            d = radius * c
            print ("Distance from RSS:", d)
            #return d


            #Change the starting node after each iteration

            for row in OurPOD:
                x = row[3]
                y = row[2]
                addr = row[1]
                # print(x)
                x = float(x)
                y = float(y)
                destination = [x, y]
                lat1, lon1 = RSS
                lat2, lon2 = destination
                new_source = lat2,lon2
                #lat1, lon1 = new_source
                lat2, lon2 = destination
                dlat = math.radians(lat1 - lat2)
                dlon = math.radians(lon1 - lon2)
                a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
                    * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                dist = radius * c

                cum_dist = cum_dist + dist
                #Assume average speed = 55 miles/hr
                speed =  55

                print("Cumulative route distance through nearest neighbor:", cum_dist)
                time = cum_dist/speed
                #for i in range(route_count):
                #cluster[i] = []

            def constraints ():
                if time < allowedtime:
                    cluster1.append([addr, x, y])
                    nodes1.append(addr)


                else:

                    lst = []
                    #for i in range(10):
                    new_cluster+=1
                    #qry = input()
                    lst += [new_cluster]
                    lst.append([addr, x, y])
                    nodes2.append(addr)

                for item in lst:
                    constraints()

    print ("PODs in cluster1:")
    for item in cluster1:
        print (item)
        print (item, sep=' >>', end='', flush=True)
    for addr in nodes1:
        print (addr)
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
distance(RSS, destination = readdata())