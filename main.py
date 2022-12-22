import networkx as nx
from cdt.causality.graph import SAM
from cdt.data import load_dataset
import matplotlib.pyplot as plt
data, graph = load_dataset("sachs")
obj = SAM()


output = obj.predict(data)    #No graph provided as an argument

print("test")

# output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
#
# output = obj.predict(data, graph)  #With a directed graph

#To view the graph created, run the below commands:

nx.draw_networkx(output, font_size=8)

plt.show()