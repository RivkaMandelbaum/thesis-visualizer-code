import networkx as nx
from pyvis.network import Network
from psynet.command_line import populate_db_from_zip_file
from dallinger.models import Node

def main():
    populate_db_from_zip_file("/Users/rivkamandelbaum/Desktop/psynet-work/visualizer/test_zipping.zip")

    nodes = Node.query.all()
    print(len(nodes))

    G = nx.complete_graph(3)
    nt = Network('500px', '500px')
    nt.from_nx(G)
    nt.show('nx.html')

if __name__ == "__main__":
    main()