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
DEFAULT_INFO_SHAPE = 'box'

PATH = "../serial-reproduction-with-selection/analysis/data/rivka-necklace-rep-data/psynet/data/"

#-----------------------------------------------------------------------
#-------------------------  Global variables  --------------------------
node_data_by_trial_maker = {} # TODO fix
info_data_by_trial_maker = {} # TODO fix
global_pos = None #TODO fix
processing_done = False

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

def to_graph_id(id, is_info):
    """ Convert an integer node or info id into a string that can be used
    to uniquely identify the graph node, by prepending 'n' or 'i'.
    Node ids and info ids can overlap, so this is necessary to differentiate them.
    Arguments: id (int), is_info (bool)
    Returns: id string
    """
    if is_info:
        return 'i' + str(id)
    else:
        return 'n' + str(id)

def from_graph_id(graph_id):
    """ Convert a graph id string (id with 'n' or 'i' prepended) to an integer
    that can be used to find the id in the Dataframes and a boolean indicating
    whether the id was for a node or an info.
    If passed an integer without prepended string, returns (id, None). If passed 'undefined', returns ('', None)
    Arguments: graph_id string
    Returns: id int, is_info bool or None
    """
    try:
        id = int(graph_id)
        return (id, None)
    except ValueError:
        if graph_id == 'undefined':
            return ('', None)

        id = int(graph_id[1:])
        if graph_id[0] == 'i':
            return (id, True)
        elif graph_id[0] == 'n':
            return (id, False)


def generate_graph(degree, trial_maker_id):
    ''' Given a degree (int or float) and a trial_maker_id in the experiment, return a DiGraph containing the nodes (with metadata from infos) and edges in that degree and associated with that trial_maker_id.
    '''
    # validation: ensure degree is a float
    if not isinstance(degree, float):
        try:
            degree = float(degree)
        except Exception as ex:
            raise Exception("When converting degree to float, the following exception occured: " + str(ex))

    # validation: ensure trial_maker_id is valid
    if trial_maker_id not in node_data_by_trial_maker.keys():
        raise Exception("Invalid trial_maker_id.")

    # use correct data for that trial_maker_id
    node_data = node_data_by_trial_maker[trial_maker_id]
    info_data = info_data_by_trial_maker[trial_maker_id]

    # create graph
    G = nx.DiGraph()

    # add nodes from node_data to the Graph, and associated infos
    # nodes are identified by their node_id
    # nodes have named attributes vertex_id and degree
    deg_nodes = node_data[node_data["degree"] == degree]
    for node_id in deg_nodes["id"].values.tolist():
        # extract vertex id
        vert_id = deg_nodes[deg_nodes["id"] == node_id]["vertex_id"].values[0]

        # color failed nodes
        node_color = DEFAULT_COLOR if (deg_nodes[deg_nodes["id"] == node_id]["failed"].values[0] == "f") else FAILED_COLOR

        # add node to graph
        G.add_node(
            to_graph_id(node_id, False),
            color=node_color,
            degree=degree,
            is_info=False,
            label=create_label(node_id),
            labelHighlightBold=True,
            shape=DEFAULT_NODE_SHAPE,
            vertex_id=vert_id
            )

        # add infos, and edges to infos
        node_infos_data = info_data[info_data["origin_id"] == node_id]

        # process into compatible types
        if len(node_infos_data) == 1:
            node_infos = [(None, node_infos_data)]
        else:
            node_infos = node_infos_data.iterrows()

        # add the actual infos
        for _, info in node_infos:
            info_id = int(info["id"])
            is_info=True

            G.add_node(
                to_graph_id(info_id, is_info),
                color=node_color,
                degree=degree,
                is_info=True,
                label=create_label(info_id),
                labelHighlightBold=True,
                origin_id=node_id,
                shape=DEFAULT_INFO_SHAPE,
                vertex_id=vert_id
            )
            G.add_edge(to_graph_id(node_id, False), to_graph_id(info_id, is_info))

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
        is_info = False
        edge_list = [(to_graph_id(int(dependent_node), is_info), to_graph_id(int(node_id), is_info)) for dependent_node in dependent_nodes]
        G.add_edges_from(edge_list)


    return G

def get_content(exp, id):
    # validation
    if exp not in node_data_by_trial_maker.keys():
        return "An error has occurred. Content cannot be displayed."

    if id in [None, '']:
        return "No content to display."

    graph_id, is_info = from_graph_id(id)

    if not is_info:
        data = node_data_by_trial_maker[exp]
    else:
        data = info_data_by_trial_maker[exp]

    string_data = data[data["id"] == graph_id].squeeze().to_json()

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
    global processing_done #TODO: Development only!
    if not processing_done:
        process_data()
        processing_done = True

    clicked_node = request.args.get('clicked-node')

    # find the correct 'exp' (trial maker id)
    exp = request.args.get('trial-maker-id')
    if exp is None or exp not in node_data_by_trial_maker.keys():
        exp = list(node_data_by_trial_maker.keys())[0]

    # find the correct 'degree'
    degree = request.args.get('degree')
    if degree is None:
        degree_cookie = request.cookies.get('degree')
        if degree_cookie is None:
            degree = node_data_by_trial_maker[exp]["degree"].min()
        else:
            degree = float(degree_cookie)


    # create network
    pyvis_net = Network(directed=True)
    nx_graph = generate_graph(degree, exp)

    # set up global network layout (fixed across degrees)
    global global_pos
    if global_pos is None:
        # print("Setting global position")
        global_pos = {}

        # get the networkx graph-id-mapped position dict
        pos = nx.spring_layout(nx_graph)

        # convert to vertex-id-mapped position dict, with only node positions added
        vertex_id_map = nx_graph.nodes(data='vertex_id')
        for graph_id, xy in pos.items():
            v_id = int(vertex_id_map[graph_id])
            if graph_id[0] == 'n':
                global_pos[v_id] = {'x': xy[0] , 'y': xy[1]}

    # read networkx graph into pyvis, add necessary attributes
    pyvis_net.from_nx(nx_graph)
    for (graph_id, node) in pyvis_net.node_map.items():
        node['title'] = str(node['label'])
        v_id = node['vertex_id']
        node['x'] = global_pos[v_id]['x'] * 10 # scaling necessary for x,y position to work
        node['y'] = global_pos[v_id]['y'] * 10

        if str(graph_id) == str(clicked_node):
            node['color'] = CLICKED_COLOR

    # generate values for the template
    graph_html = pyvis_net.generate_html()

    script_to_replace = 'network = new vis.Network(container, data, options);'
    click_script = 'network = new vis.Network(container, data, options);\
        network.on("click", function(properties) {\
            let node_id = properties.nodes[0];\
            console.log(node_id);\
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
        content=get_content(exp, clicked_node)
        )

    response = make_response(page_html)

    response.set_cookie('degree', str(degree))
    response.set_cookie('exp', exp)

    return response