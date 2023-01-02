import networkx as nx
import pandas as pd
from pyvis.network import Network
import sys

def main():
    if len(sys.argv) != 2:
        print("usage: python3 upgraded_visualizer.py [path to CSV]")

    # import data
    csv_path = sys.argv[1]

    # read Nodes CSV into Dataframe
    nodes = pd.read_csv(csv_path)

    # Convert to slices of Nodes by degree
    node_data = nodes
    node_data = node_data[nodes["type"] == "graph_chain_node"]
    node_data = node_data[node_data["failed"] == "f"]
    node_data = node_data[["id", "network_id", "degree", "definition", "seed", "vertex_id", "dependent_vertex_ids"]]

    first_slice = node_data[node_data["degree"] == 1.0]

    # Convert slices to Networkx objects


    # Basic visulazation of a single slice

    # G = nx.complete_graph(3)
    # nt = Network('500px', '500px')
    # nt.from_nx(G)
    # nt.show('nx.html')

if __name__ == "__main__":
    main()