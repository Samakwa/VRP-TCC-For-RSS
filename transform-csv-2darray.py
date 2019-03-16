
import csv


#with open('Route_Distances2.csv','rb') as csvfile:
 #   data = list(csv.reader(csvfile))

#print(data)

# find_num_vehicles calculate the number of vehicles need to cover all root with a limited distance
# input: dictionary of routes and distances
# it take the dictionary of routes and distances
# return the name of route with minimal distance
def find_shortest_distance(route):

    if bool(route) is not True:
        return None, 0
    #print "!!!!!!!",route
    # choose the first item as the minmum
    min_node = route[0].keys()[0]
    min_dstnce = route[0].values()[0]
    for r in route:
        #print r.values()[0], type(r)
        dstnce = r.values()[0]

        if dstnce < min_dstnce:
            min_dstnce = dstnce
            min_node = r.keys()[0]

    return min_node, min_dstnce

def remove_internal_node(node,route_distance_dict ):
   # print "before removing!!!",route_distance_dict
    for k,v in route_distance_dict.items():

        route_distance_dict[k] = [item for item in v if item.keys()[0] != node]
   # print "after removing!!!!!",route_distance_dict

    return


def find_num_vehicles(route_distance_dict, max_distance):
    limited_distance = 0
    route_name = []
    track_rout = []
    all_routes = []

    # the input dataset is dictionary of source routes as key and list of destination with their distance
    # for example 'RSS': [{'Clear Brook High School': 250.2700892}, {'Pershing Middle School': 80.67111667}, {'Sharpstown Middle School': 100.00564}, {'Waller ISD Stadium': 330.9867553}]
    # we need to start from RSS so we need to find RSS in the dictionary
    source_node_name = 'RSS'
    track_rout.append(source_node_name)
    print (max_distance)
    # check that the maximum distance in the dataset is smaller that maximum distance
    max_distance = 94
    # the first vichel that we need to start
    num_vhcls = 1
    rss_dict = {}
    source_node = route_distance_dict.get(source_node_name)
    rss_dict = {source_node_name: source_node}
    while bool(route_distance_dict):
     #find the dictionary of each node which contains its distance from all other nodes

     source_node = route_distance_dict.get(source_node_name)
     #print "source node", source_node
     #find the shortest distance
     destination_node_name, dstnc = find_shortest_distance(source_node)
     print ("destination node name",destination_node_name, dstnc)

     if destination_node_name == None and dstnc == 0:
         #route_distance_dict.pop(source_node_name)
         #print "this is the last one", source_node_name, route_distance_dict
         #continue
         break
     #print "next source node is", destination_node_name, dstnc
     limited_distance += dstnc

     #print "accumulated distance:", limited_distance
     if limited_distance <= max_distance:
         track_rout.append(destination_node_name)
         #if the source node is RSS we shouldn't remove it from dictionary
         #because after reaching the maximum distance we need to start from RSS again
         print ("source node name",source_node_name, limited_distance)
         print
         if source_node_name in route_distance_dict.keys():
             route_distance_dict.pop(source_node_name)
             remove_internal_node(source_node_name, route_distance_dict)
             remove_internal_node(source_node_name, rss_dict)


         #print "after removing!!!",route_distance_dict
         #print "after removing rss!!!",rss_dict
         #source_node_name = destination_node_name
        # print "each_rout", track_rout, route_distance_dict
     elif limited_distance > max_distance:

         num_vhcls += 1
         all_routes.append(track_rout)
         track_rout = []
         print ("source and destination and limited distance after reaching the maximum disatance",source_node_name, destination_node_name, limited_distance)
         if source_node_name in route_distance_dict.keys():
            route_distance_dict.pop(source_node_name)
            remove_internal_node(source_node_name, route_distance_dict)
            remove_internal_node(source_node_name, rss_dict)


         limited_distance = 0
         source_node_name = 'RSS'
         track_rout.append(source_node_name)
         track_rout.append(destination_node_name)
     source_node_name = destination_node_name
    if bool(track_rout):
        print ("last track", track_rout)
        all_routes.append(track_rout)
        track_rout = []
    print ("all routes",all_routes, track_rout, limited_distance, max_distance)
    print ("length of rout", len(all_routes))
    print ("the number of needed vehicles", num_vhcls)

    return num_vhcls, all_routes
#print the determined routes
def print_routs (nmbr_vchl, routes):
    for i in range(nmbr_vchl):
            print (routes[i])
def readcsv(filename):
    #ifile = open(filename, "rU")

    #reader = csv.reader(ifile)


    f = open(filename, "r+")
    csv_f = csv.reader(f)
    data = {}
    source_pod = {}
    source_pod.update({'RSS': 0})
    matrix = []
    source_POD_old = '0'
    source_name = 'RSS'
    ind = 0
    for row in csv_f:


        if row[0] == source_POD_old and row[1] != "Source _POD_Name":
           dest_pod = {}
           dest_pod.update({row[4]: float(row[5])})
           matrix.append(dest_pod)


        elif row[0]!= "Source POD ID" and row[1] != "Source _POD_Name":

            data[source_name] = matrix
            matrix = []
            source_POD_old = row[0]
            source_name = row[1]
            dest_pod = {}
            dest_pod.update({row[4]: float(row[5])})
            matrix.append(dest_pod)
            ind += 1
    data[row[1]] = matrix
    print (data)

    return data
filename = 'Route_Distances2.csv'
route_distance_dict = readcsv(filename)
#print route_distance
max_speed = 57
max_time = 36
max_distance = float (max_speed * max_time)
print ("this is the maximum distance for each vehicle", max_distance)

nmbr_vchl, routes = find_num_vehicles(route_distance_dict, max_distance)
print_routs (nmbr_vchl, routes)