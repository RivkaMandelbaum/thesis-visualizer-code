#-----------------------------------------------------------------------
# upgraded_visualizer.py
# Author: Rivka Mandelbaum
#-----------------------------------------------------------------------
#--------------------------   Imports  ---------------------------------

import pandas as pd
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template, make_response

#-----------------------------------------------------------------------
#-------------------------- Constants ----------------------------------
DEFAULT_COLOR = '#97c2fc' # from PyVis
FAILED_COLOR = 'red'

PATH = "../serial-reproduction-with-selection/analysis/data/rivka-necklace-rep-data/psynet/data/"

#-----------------------------------------------------------------------
#-------------------------  Global variables  --------------------------
node_data_by_trial_maker = {} # TODO fix
info_data_by_trial_maker = {} # TODO fix

app = Flask(__name__, template_folder='./templates')

#-----------------------------------------------------------------------
#----------------------------  Functions  ------------------------------
def process_data():
    nodes = pd.read_csv(PATH + "node.csv", low_memory=False)
    networks = pd.read_csv(PATH + "network.csv", low_memory=False)
    infos = pd.read_csv(PATH + "info.csv")

   # filter networks: role = experiment
    network_data = networks
    network_data = network_data[network_data["role"] == "experiment"]

    # filter per trial maker ID
    trial_maker_ids = network_data["trial_maker_id"].unique()

    for trial_maker_id in trial_maker_ids:
        network_data = network_data[network_data["trial_maker_id"] == trial_maker_id]

        experiment_network_ids = list(network_data['id'].to_numpy())

        # filter nodes; sort
        node_data = nodes
        node_data = node_data[nodes["type"] == "graph_chain_node"]
        node_data = node_data[node_data["network_id"].isin(experiment_network_ids)]
        node_data = node_data[["id", "network_id", "degree", "definition", "seed", "vertex_id", "dependent_vertex_ids", "failed"]]
        node_data = node_data.sort_values(["network_id", "degree"])

        # filter infos like nodes, sort
        info_data = infos
        info_data = info_data[infos["type"] == "graph_chain_trial"]
        info_data = info_data[["id", "creation_time", "details", "origin_id", "network_id", "participant_id", "failed"]] # TODO: Probably want more columns here

        info_data = info_data.sort_values(["network_id", "origin_id"])

        # add to the dictionary
        node_data_by_trial_maker[trial_maker_id] = node_data
        info_data_by_trial_maker[trial_maker_id] = info_data

def generate_graph(degree, trial_maker_id):
    ''' Given a degree (int or float) and a trial_maker_id in the experiment, return a DiGraph containing the nodes (with metadata from infos) and edges in that degree and associated with that trial_maker_id.
    '''
    # validation: ensure degree is a float
    if not isinstance(degree, float):
        try:
            degree = float(degree)
        except Exception as ex:
            raise("When converting degree to float, the following exception occured: " + str(ex))

    # validation: ensure trial_maker_id is valid
    if trial_maker_id not in node_data_by_trial_maker.keys():
        print(node_data_by_trial_maker)
        raise Exception("Invalid trial_maker_id.")

    # use correct data for that trial_maker_id
    node_data = node_data_by_trial_maker[trial_maker_id]
    info_data = info_data_by_trial_maker[trial_maker_id]

    # create graph
    G = nx.DiGraph()

    # add nodes from node_data to the Graph
    # nodes are identified by their node_id
    # nodes have named attributes vertex_id and degree
    deg_nodes = node_data[node_data["degree"] == degree]
    for node_id in deg_nodes["id"].values.tolist():
        # extract vertex id
        vert_id = deg_nodes[deg_nodes["id"] == node_id]["vertex_id"].values[0]

        # add metadata from infos
        info_row = info_data[info_data["origin_id"] == node_id]
        try:
            creation_time = info_row["creation_time"].values[0]
        except:
            creation_time = None

        # color failed nodes
        node_color = DEFAULT_COLOR if (deg_nodes[deg_nodes["id"] == node_id]["failed"].values[0] == "f") else FAILED_COLOR

        G.add_node(node_id, vertex_id=vert_id, degree=degree, creation_time=creation_time, color=node_color, label=str(int(vert_id)))

    # add edges to the Graph: iterate over deg_nodes, add incoming edges
    # using dependent_vertex_ids column
    for _, ser in deg_nodes.iterrows():
        node_id = ser["id"]

        # get dependent vertices (incoming edges) in a list
        dependent_vertices = ser["dependent_vertex_ids"].strip('][').split(',')

        # find the corresponding row of deg_nodes for each vertex_id
        dependent_nodes = [deg_nodes[deg_nodes["vertex_id"] == float(v)] for v in dependent_vertices]

        # extract the node_id from each dependent vertex row
        dependent_nodes = [n["id"].values[0] for n in dependent_nodes]

        # add as edges (dependent node --> curent node)
        edge_list = [(int(dependent_node), int(node_id)) for dependent_node in dependent_nodes]
        G.add_edges_from(edge_list)

    return G

@app.route('/')
@app.route('/index')
def create_visualizer():

    # process data into dicts (global variables)
    process_data()


    pyvis_net = Network(directed=True)
    pyvis_net.from_nx(generate_graph(1.0, "graph_experiment"))

    for node in pyvis_net.nodes:
        node['title'] = node['label']

    graph_html = pyvis_net.generate_html()
    min_degree = node_data_by_trial_maker["graph_experiment"]["degree"].min()
    max_degree = node_data_by_trial_maker["graph_experiment"]["degree"].max()

    page_html = render_template(
        'dashboard_visualizer.html',
        graph=graph_html,
        trialmaker_options=node_data_by_trial_maker.keys(),
        degree_min=min_degree,
        degree_max=max_degree,
        physics_options=["barnes hut", "placeholder 1"]
        )

    response = make_response(page_html)
    return response