import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community.community_louvain as community_louvain



#build from edges 
#the k factor expresses how strong the gravity is between the nodes. A larger k means nodes are more separated, so less "structure" is seen in the graph.
graph = nx.from_pandas_edgelist(DF[['Index_x','Index_y','logFlightCount']], 'Index_x', 'Index_y', 'logFlightCount')
edges,weights = zip(*nx.get_edge_attributes(graph,'logFlightCount').items())
plt.figure(figsize=(50,30))
nx.draw_networkx(graph, pos=nx.spring_layout(graph,weight='logFlightCount'), with_labels=True, edge_color=weights, edge_cmap=plt.cm.autumn_r, font_size=25, node_size=1500, node_color='#c0d6e4')
plt.show()
plt.figure(figsize=(50,30))
nx.draw_networkx(graph, pos=nx.spring_layout(graph, k=1.5,weight='logFlightCount'), with_labels=True, edge_color=weights, edge_cmap=plt.cm.autumn_r, font_size=25, node_size=1500, node_color='#c0d6e4')
plt.show()

#spectrar decomposition (a process that has some parallels to PCA)
plt.figure(figsize=(50,30))
nx.draw_networkx(graph, pos=nx.spectral_layout(graph, scale=10, dim=2,weight='logFlightCount'), with_labels=True, edge_color=weights, edge_cmap=plt.cm.autumn_r, font_size=25, node_size=1500, node_color='#c0d6e4')
plt.show()

#degree of centrality
deg_cent = nx.degree_centrality(graph)
cent_array = np.fromiter(deg_cent.values(), float)
pd.DataFrame(pd.Series(deg_cent) ).sort_values(0, ascending=False)

#finding communities
part = community_louvain.best_partition(graph)
values = [part.get(node) for node in graph.nodes()]
plt.figure(figsize=(50,30))
nx.draw_networkx(graph, pos=nx.spring_layout(graph, k=0.01,weight='logFlightCount'), cmap = plt.get_cmap('Set3'), node_color = values, with_labels=True, edge_color=weights, edge_cmap=plt.cm.autumn_r, font_size=25, node_size=1500)
plt.show()