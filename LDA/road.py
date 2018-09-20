import osmnx as ox, networkx as nx, matplotlib.cm as cm, pandas as pd, numpy as np
ox.config(log_file=True, log_console=True, use_cache=True)

query_point = (37.8, -122.4)
dest_point = (38.6, -121.3)

G = ox.graph_from_point(query_point, distance=500, distance_type='network', network_type='walk')
origin_node = ox.get_nearest_node(G, query_point)
destination_node = ox.get_nearest_node(G, dest_point)
# find the route between these nodes then plot it
route = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
#fig, ax = ox.plot_graph_route(G, route)
print (route)
#print origin_node