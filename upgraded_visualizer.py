#-----------------------------------------------------------------------
# upgraded_visualizer.py
# Author: Rivka Mandelbaum
#-----------------------------------------------------------------------
#--------------------------   Imports  ---------------------------------

import pandas as pd
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template, make_response, request

#-----------------------------------------------------------------------
#-------------------------- Constants ----------------------------------
DEFAULT_COLOR = '#97c2fc' # from PyVis
FAILED_COLOR = 'red'
CLICKED_COLOR = 'blue'

DEFAULT_NODE_SHAPE = 'circle' # puts label inside node

PATH = "../serial-reproduction-with-selection/analysis/data/rivka-necklace-rep-data/psynet/data/"

#-----------------------------------------------------------------------
#-------------------------  Global variables  --------------------------
node_data_by_trial_maker = {} # TODO fix
info_data_by_trial_maker = {} # TODO fix
global_pos = None #TODO fix

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

        G.add_node(node_id, vertex_id=vert_id, degree=degree, creation_time=creation_time, color=node_color, label=create_label(node_id), shape=DEFAULT_NODE_SHAPE, labelHighlightBold=True)

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

def get_node_content(exp, node_id):
    # validation
    if exp not in node_data_by_trial_maker.keys():
        return "An error has occurred. Content cannot be displayed."

    if node_id in [None, '']:
        return "No content to display."

    node_data = node_data_by_trial_maker[exp]
    string_data = node_data[node_data["id"] == int(node_id)].squeeze().to_json()

    return string_data

def create_label(id):
    # this sucks TODO
    try:
        label = '    %s    ' % str(int(id))
    except ValueError:
        label = '    %s    ' % str(id)

    return label

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def create_visualizer():
    # process data into dicts (global variables)
    process_data()

    clicked_node = request.args.get('clicked-node')

    # find the correct 'exp' (trial maker id)
    exp = request.args.get('trial-maker-id')
    if exp is None or exp not in node_data_by_trial_maker.keys():
        exp = list(node_data_by_trial_maker.keys())[0]

    # find the correct 'degree'
    degree = request.args.get('degree')
    if degree is None:
        degree = node_data_by_trial_maker[exp]["degree"].min()

    # create network
    pyvis_net = Network(directed=True)
    nx_graph = generate_graph(degree, exp)

    # set up global network layout (fixed across degrees)
    global global_pos
    if global_pos is None:
        global_pos = {}

        # get the networkx node-id-mapped position dict
        pos = nx.spring_layout(nx_graph)

        # convert to vertex-id-mapped position dict
        vertex_id_map = nx_graph.nodes(data='vertex_id')
        for n_id, xy in pos.items():
            v_id = vertex_id_map[n_id]
            global_pos[v_id] = {'x': xy[0] , 'y': xy[1]}

    # read networkx graph into pyvis, add necessary attributes
    pyvis_net.from_nx(nx_graph)
    for (n_id, node) in pyvis_net.node_map.items():
        node['title'] = node['label']
        v_id = node['vertex_id']
        node['x'] = global_pos[v_id]['x'] * 10 # scaling necessary for x,y position to work
        node['y'] = global_pos[v_id]['y'] * 10

        if str(n_id) == str(clicked_node):
            node['color'] = CLICKED_COLOR

    # generate values for the template
    graph_html = pyvis_net.generate_html()

    script_to_replace = 'network = new vis.Network(container, data, options);'
    click_script = 'network = new vis.Network(container, data, options);\
        network.on("click", function(properties) {\
            let node_id = properties.nodes[0];\
            node_form = document.getElementById("clicked-node-form");\
            node_form_input = document.getElementById("clicked-node-input");\
            node_form_input.value = node_id;\
            node_form.submit();\
        })'

    graph_html = graph_html.replace(script_to_replace, click_script)

    node_data = node_data_by_trial_maker[exp]
    min_degree = node_data["degree"].min()
    max_degree = node_data["degree"].max()
    min_vertex_id = node_data["vertex_id"].min()
    max_vertex_id = node_data["vertex_id"].max()

    trialmaker_options = node_data_by_trial_maker.keys()

    # render template and return response
    page_html = render_template(
        'dashboard_visualizer.html',
        graph=graph_html,
        trialmaker_options=trialmaker_options,
        degree_min=min_degree,
        degree_max=max_degree,
        degree_placeholder=degree,
        physics_options=["barnes hut", "placeholder 1"],
        find_min=min_vertex_id,
        find_max=max_vertex_id,
        content=get_node_content(exp, clicked_node)
        )

    response = make_response(page_html)
    return response