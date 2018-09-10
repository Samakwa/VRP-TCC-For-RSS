import math
import csv


allowedtime= 48
cluster1 =[]
cluster2 =[]
nodes1=[]
nodes2 =[]

#origin = input ( "Enter Cordinates of Origin; lat, long: ")
origin = (29.779630, -95.436960)
def readdata():
    with open('test1.csv', 'r+') as in_file:
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


def distance(origin, destination):
    #readdata()
    with open('test1.csv', 'r+') as in_file:
        OurPOD = csv.reader(in_file)
        has_header = csv.Sniffer().has_header(in_file.readline())
        in_file.seek(0)  # Rewind.
        if has_header:
            next(OurPOD)  # Skip header row.

        for row in OurPOD:
            x = row[3]
            y = row[2]
            # print(x)
            x = float(x)
            y = float(y)
            destination = [x,y]
            lat1, lon1 = origin
            lat2, lon2 = destination
            origin = (29.779630, -95.436960)
            radius = 3959 # Miles

            dlat = math.radians(lat2-lat1)
            dlon = math.radians(lon2-lon1)
            a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            d = radius * c
            print (d)
            #return d

            cum_dist = 0
            """
            for k, v in nodelist.items():
                cum_dist = cum_dist + v
                print(cum_dist)
                if cum_dist < allowedtime:
                    cluster1.append([k, v])
                    nodes1.append(k)


                else:
                    cluster2.append([k, v])
                    nodes2.append(k)
            """

distance(origin, destination = readdata())