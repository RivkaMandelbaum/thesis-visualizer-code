#-----------------------------------------------------------------------
# upgraded_visualizer.py
# Author: Rivka Mandelbaum
#-----------------------------------------------------------------------
#--------------------------   Imports  ---------------------------------

import pandas as pd
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template, make_response, request
from numpy import True_, random
#-----------------------------------------------------------------------
#-------------------------- Constants ----------------------------------
DEFAULT_COLOR = '#97c2fc' # from PyVis
FAILED_COLOR = 'red'
CLICKED_COLOR = '#0000FF' # dark blue
NEIGHBOR_COLOR = '#4C61FE' # midpoint between default color and dark blue

DEFAULT_NODE_SHAPE = 'circle' # puts label inside node
DEFAULT_INFO_SHAPE = 'box'

# columns to remove from infos
COLS_TO_DROP = ["property1", "property2", "property3", "property4", "property5"]

# constants for the 'show X node only' settings
SHOW_NODES_ALL = 'all'
SHOW_NODES_INCOMING = 'incoming'
SHOW_NODES_OUTGOING = 'outgoing'
SHOW_NODES_CONNECTED = 'connected'
SHOW_OPTION = 'show-option'

# solver
BARNES_HUT = 'barnes-hut'
FORCE_ATLAS_2BASED = 'force-atlas'
REPULSION = 'repulsion'
HIERARCHICAL_REPULSION = 'hrepulsion'
VALID_SOLVERS = {
    BARNES_HUT: "Barnes Hut",
    FORCE_ATLAS_2BASED: "forceAtlas2Based",
    REPULSION: "Repulsion",
    HIERARCHICAL_REPULSION: "Hierarchical Repulsion"
}

# layout options
LAYOUT_OPTIONS = {
    'circular': {'name': 'Circular Layout', 'has_seed': False, 'func': nx.circular_layout, 'scale': 500},
    'kamada-kawai': {'name': 'Kamada-Kawai Layout', 'has_seed': False, 'func': nx.kamada_kawai_layout, 'scale': 1500},
    'random': {'name': 'Random Layout', 'has_seed': True, 'func': nx.random_layout, 'scale': 1500},
    'shell': {'name': 'Shell Layout', 'has_seed': False, 'func': nx.shell_layout, 'scale': 500},
    'spectral': {'name': 'Spectral Layout', 'has_seed': False, 'func': nx.spectral_layout, 'scale': 1500},
    'spiral': {'name': 'Spiral Layout', 'has_seed': False, 'func': nx.spiral_layout, 'scale': 1000},
    'spring': {'name': 'Spring Layout', 'has_seed': True, 'func': nx.spring_layout, 'scale': 1500}
}
DEFAULT_LAYOUT = 'spring' # networkx default
# PATH = app.config.get('data_path') #"../serial-reproduction-with-selection/analysis/data/rivka-necklace-rep-data/psynet/data/"

# settings dict
CLICKED_NODE = 'clicked-node'
DEGREE = 'degree'
EXP = 'exp'
LAYOUT = 'layout'
SEED = 'seed'
SOLVER = 'solver'
SHOW_INFOS = 'show-infos'
SHOW_INCOMING = 'incoming'
SHOW_OUTGOING = 'outgoing'
GRAPH_SETTINGS = [CLICKED_NODE, EXP, DEGREE, SHOW_INFOS, SHOW_OUTGOING, SHOW_INCOMING, SOLVER, SEED, LAYOUT]

#-----------------------------------------------------------------------
#-------------------------  Global variables  --------------------------
node_data_by_trial_maker = {} # TODO fix
info_data_by_trial_maker = {} # TODO fix
vertex_pos = None #TODO fix
processing_done = False

class ClickedNodeException(Exception):
    "Clicked node or info not present in data"
    pass

app = Flask(__name__, template_folder='./templates')

#-----------------------------------------------------------------------
#----------------------------  Functions  ------------------------------
#-----------------------------------------------------------------------

#---------------------- Simple helper functions --------------------------
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
    If passed an integer without prepended string, returns (id, None). If passed 'undefined' or None, returns ('', None)
    Arguments: graph_id string
    Returns: id int, is_info bool or None
    """
    try:
        id = int(graph_id)
        return (id, None)
    except:
        if graph_id in ['undefined', None, '']:
            return ('', None)

        id = int(graph_id[1:])
        if graph_id[0] == 'i':
            return (id, True)
        elif graph_id[0] == 'n':
            return (id, False)

def get_content_list(exp, id):
    ''' Get the content of a node/info in a given trial_maker_id to display
    in the content box. Return as array of strings. Each array element will be displayed on its own line.
    '''
    # validation
    if exp not in node_data_by_trial_maker.keys():
        return ["An error has occurred. Try reloading the page."]
    if id in [None, '']:
        return ["No content to display."]

    # get the content
    graph_id, is_info = from_graph_id(id)

    if not is_info:
        data = node_data_by_trial_maker[exp]
    else:
        data = info_data_by_trial_maker[exp]

    # format content and return
    dict_data = data[data["id"] == graph_id].squeeze().to_dict()
    content_strings = []
    for (key, val) in dict_data.items():
        content_strings.append('%s: %s' % (key, str(val)))

    return content_strings

def create_label(id):
    ''' Create a label for node/info with given id. The label goes inside
    the node/info when the graph is rendered.
    '''
    # this sucks TODO
    try:
        label = '    %s    ' % str(int(id))
    except ValueError:
        label = '    %s    ' % str(id)

    return label

#---------------------- Complex helper functions --------------------------
def process_data(path):
    """ Reads the CSVs produced by exporting data into data structures
    that can be used by the visualizer. Specifically, fills in global
    dicts node_data_by_trial_maker and info_data_by_trial_maker, so that
    each one has trial_maker_id:Dataframe pairs. node_data_by_trial_maker
    contains filtered data from node.csv, and info_data_by_trial_maker
    contains filtered data from info.csv.
    """
    global processing_done #TODO: Development only!
    if processing_done:
        return

    # read CSVs
    nodes = pd.read_csv(path + "node.csv", low_memory=False)
    networks = pd.read_csv(path + "network.csv", low_memory=False)
    infos = pd.read_csv(path + "info.csv")

   # filter networks: role = experiment
    network_data = networks
    network_data = network_data[network_data["role"] == "experiment"]

    # fill in node data and info data, for each trial_maker_id
    trial_maker_ids = network_data["trial_maker_id"].unique()

    for trial_maker_id in trial_maker_ids:
        # find relevant network ids for this trial_maker_id
        network_data = network_data[network_data["trial_maker_id"] == trial_maker_id]
        experiment_network_ids = list(network_data['id'].to_numpy())

        # filter and sort nodes
        node_data = nodes
        node_data = node_data[nodes["type"] == "graph_chain_node"] #TODO generalizable?
        node_data = node_data[node_data["network_id"].isin(experiment_network_ids)]
        node_data = node_data[["id", "network_id", "degree", "definition", "seed", "vertex_id", "dependent_vertex_ids", "failed"]]
        node_data = node_data.sort_values(["network_id", "degree"])

        # filter infos like nodes
        info_data = infos
        info_data = info_data[infos["type"] == "graph_chain_trial"]
        try:
            info_data = info_data.drop(COLS_TO_DROP, axis="columns")
        except KeyError:
            print("Data does not contain property1-property5")

        # add filtered Dataframes to the global dicts
        node_data_by_trial_maker[trial_maker_id] = node_data
        info_data_by_trial_maker[trial_maker_id] = info_data

    processing_done = True

def update_clicked_node(graph_id, degree, trial_maker_id):
    # the point of this is to update stale clicked nodes
    # if the node id is "" there was either an error or nothing has been set
    if graph_id == "":
        return ""

    clicked_id, clicked_is_info = from_graph_id(graph_id)
    trial_data = node_data_by_trial_maker
    if clicked_is_info: # treat infos as their parent nodes
        try:
            info_data = info_data_by_trial_maker[trial_maker_id]
            info = info_data[info_data["id"] == clicked_id]
            clicked_id = info["origin_id"].values[0]
        except:
            print("Failed to convert info to parent node in update_clicked_node.")
            return ""

    if trial_maker_id not in trial_data.keys():
        return ""
    else:
        trial_data = trial_data[trial_maker_id]

    try:
        degree = float(degree)
        node = trial_data[trial_data["id"] == clicked_id]
        if node["degree"].values[0] == degree: #TODO tmid validation
            # the given graph_id belongs to the current degree and does
            # not need to be updated
            return graph_id

        vertex_id = float(node["vertex_id"].values[0])
        degree_nodes = trial_data[trial_data["degree"] == degree]
        new_node = degree_nodes[degree_nodes["vertex_id"] == vertex_id]
        new_node_id = int(new_node["id"].values[0])

        new_clicked_id = to_graph_id(new_node_id, False)
        return new_clicked_id
    except Exception as ex:
        print("Failed to find correct node in update_clicked_node")
        print(str(ex))
        return ""

def get_settings(request, from_index=False):
    ''' Get settings from the request, or use correct defaults.
        Must be run after process_data() so that the trial_maker_id validation works properly.
        Argument: request
        Returns: dictionary of:
            clicked_node (graph ID)
            exp (string)
            degree (float)
            layout (string)
            show_infos (bool)
            show_outgoing (bool)
            show_incoming (bool)
            solver (string)
            seed (int)
    '''
    # check that data has been processed
    if len(list(node_data_by_trial_maker.keys())) == 0:
        raise Exception("Settings cannot be found before data is processed.")

    settings = {}

    # find the correct 'exp' (trial maker id)
    exp = request.args.get('trial-maker-id')
    if exp is None or exp not in node_data_by_trial_maker.keys():
        exp = list(node_data_by_trial_maker.keys())[0]
    settings[EXP] = exp

    # find the correct 'degree' and convert to float
    degree = request.args.get(DEGREE)
    if degree in [None, '']:
        degree_cookie = request.cookies.get(DEGREE)
        if degree_cookie is None:
            degree = node_data_by_trial_maker[exp]["degree"].min()
        else:
            degree = degree_cookie
    settings[DEGREE] = float(degree)

    # get clicked node id
    clicked_node = request.args.get(CLICKED_NODE)
    if clicked_node is None:
        clicked_node = request.cookies.get(CLICKED_NODE)

    if True:
        clicked_node = update_clicked_node(clicked_node, settings[DEGREE], settings[EXP])
    settings[CLICKED_NODE] = clicked_node if (clicked_node is not None) else ''

    # check whether show infos is on, convert to boolean
    if from_index:
        show_infos = request.cookies.get(SHOW_INFOS)
    else:
        show_infos = request.args.get(SHOW_INFOS)
    show_infos = True if (show_infos == "true") else False
    settings[SHOW_INFOS] = show_infos

    # check whether show_incoming and show_outgoing should be on
    show_incoming = False
    show_outgoing = False

    show_option = request.args.get(SHOW_OPTION)
    if not show_option:
        show_option = request.cookies.get(SHOW_OPTION)
    if show_option in [SHOW_NODES_CONNECTED, SHOW_NODES_INCOMING]:
        show_incoming = True
    if show_option in [SHOW_NODES_CONNECTED, SHOW_NODES_OUTGOING]:
        show_outgoing = True

    settings[SHOW_OUTGOING] = show_outgoing
    settings[SHOW_INCOMING] = show_incoming

    # find the solver
    solver = request.args.get(SOLVER)
    if solver is None:
        solver = request.cookies.get(SOLVER)
    if solver not in VALID_SOLVERS.keys():
        solver = BARNES_HUT
    settings[SOLVER] = solver

    # find the layout (request or cookie)
    layout = request.args.get(LAYOUT)
    if not layout:
        layout = request.cookies.get(LAYOUT)
    if layout not in LAYOUT_OPTIONS.keys():
        layout = DEFAULT_LAYOUT
    settings[LAYOUT] = layout

    # get the seed (or None if it was not the setting that was changed)
    seed = request.args.get(SEED)
    if seed in ['', 'undefined']:
        seed = None
    settings[SEED] = int(seed) if seed is not None else None

    return settings

def set_graph_cookies(response, settings):
    ''' Sets cookies based on settings:
            clicked-node
            degree
            exp
            layout
            show-infos
            show-option
    '''
    response.set_cookie(CLICKED_NODE, settings[CLICKED_NODE])
    response.set_cookie(DEGREE, str(settings[DEGREE]))
    response.set_cookie('exp', settings[EXP])
    response.set_cookie(LAYOUT, settings[LAYOUT])
    response.set_cookie(SHOW_INFOS, "true" if settings[SHOW_INFOS] else "false")

    show_option_cookie = "all"
    if settings[SHOW_OUTGOING] and settings[SHOW_INCOMING]:
        show_option_cookie = SHOW_NODES_CONNECTED
    elif settings[SHOW_OUTGOING]:
        show_option_cookie = SHOW_NODES_OUTGOING
    elif settings[SHOW_INCOMING]:
        show_option_cookie = SHOW_NODES_INCOMING
    response.set_cookie(SHOW_OPTION, show_option_cookie)

    response.set_cookie(SOLVER, settings[SOLVER])

def add_node_to_networkx(G, degree, trial_maker_id, node_id, clicked_node, show_outgoing, show_incoming):
    """ Adds node to networkx DiGraph.
        Arguments:
            G: networkx DiGraph
            degree: float
            trial_maker_id: string
            node_id: int
            clicked_node: string (in graph_id format)
            show_outgoing: bool
            show_incoming: bool
        Node attributes:
            graph_id    ('n' + 'id' field)
            color
            degree
            is_info     (False)
            label       ('id' with spacing)
            labelHighlightBold (True)
            shape       (DEFAULT_NODE_SHAPE)
            vertex_id   ('vertex_id' field)
            hidden
    """
    # filter data
    node_data = node_data_by_trial_maker[trial_maker_id]
    info_data = info_data_by_trial_maker[trial_maker_id]
    clicked_graph_id, clicked_is_info = from_graph_id(clicked_node)

    # extract vertex id
    vert_id = node_data[node_data["id"] == node_id]["vertex_id"].values[0]

    # set node color and hidden status for clicked/neighbor nodes
    node_color = DEFAULT_COLOR
    node_is_hidden = True if (show_outgoing or show_incoming) else False

    # check if node was clicked on
    if to_graph_id(node_id, False) == clicked_node: # node was clicked on
        node_color = CLICKED_COLOR
        node_is_hidden = False
    # check if node's child info was clicked on
    elif clicked_is_info and info_data[info_data["id"] == clicked_graph_id]["origin_id"].values[0] == node_id: # node's info was clicked on
        node_color = CLICKED_COLOR
        node_is_hidden = False
    # check if node's neighbor was clicked on (connected either direction)
    elif clicked_graph_id != '':
        incoming_to_curr = node_data[node_data["id"] == from_graph_id(node_id)[0]]["dependent_vertex_ids"].values[0].strip('][').split(', ')
        if clicked_graph_id == '':
            print('empty string from clicked graph id')
        clicked_vertex = str(int(node_data[node_data["id"] == clicked_graph_id]["vertex_id"].values[0]))

        if clicked_vertex in incoming_to_curr:
            node_color = NEIGHBOR_COLOR
            if show_outgoing:
                node_is_hidden = False

        incoming_to_clicked = node_data[node_data["id"] == clicked_graph_id]["dependent_vertex_ids"].values[0].strip('][').split(', ')

        if str(int(vert_id)) in incoming_to_clicked:
            node_color = NEIGHBOR_COLOR
            if show_incoming:
                node_is_hidden = False

    # color failed nodes (overwrites clicked coloring if relevant)
    if (node_data[node_data["id"] == node_id]["failed"].values[0] == "t"):
        node_color = FAILED_COLOR

    # add node to graph
    G.add_node(
        to_graph_id(node_id, False),
        color=node_color,
        degree=degree,
        is_info=False,
        label=create_label(vert_id),
        labelHighlightBold=True,
        shape=DEFAULT_NODE_SHAPE,
        vertex_id=vert_id,
        hidden=node_is_hidden,
        physics=False
        )

def add_infos_to_networkx(G, degree, trial_maker_id, node_id, clicked_node, show_infos):
    """ Adds infos associated with given node_id to networkx DiGraph, and edges.
        Arguments:
            G: networkx DiGraph
            degree: float
            trial_maker_id: string
            node_id: int
            clicked_node: string (in graph_id format)
            show_infos: bool
            show_outgoing: bool
        Info attributes:
            graph_id    ('i' + 'id' field)
            color
            degree
            is_info     (True)
            label       ('id' field with spacing)
            labelHighlightBold (True)
            origin_id   (origin node's 'id' field)
            shape       (DEFAULT_INFO_SHAPE)
            vertex_id   (origin node's 'vertex_id' field)
            hidden      (bool, depends on settings)
    """
    # get associated info data
    node_data = node_data_by_trial_maker[trial_maker_id]
    info_data = info_data_by_trial_maker[trial_maker_id]
    node_infos_data = info_data[info_data["origin_id"] == node_id]

    if len(node_infos_data) == 1: # process to make compatible format
        node_infos = [(None, node_infos_data)]
    else:
        node_infos = node_infos_data.iterrows()

    # add the actual infos
    for _, info in node_infos:
        info_id = int(info["id"])

        is_info=True

        # set info visibility (hidden by default)
        info_is_hidden = True
        if show_infos and not G.nodes[to_graph_id(node_id, False)]["hidden"]:
            # this info was clicked on; this info's parent was clicked on;
            # or a neighbor of this info's parent was clicked on
            if to_graph_id(node_id, False) == clicked_node or \
                to_graph_id(info_id, True) == clicked_node or \
                clicked_node in G.pred[to_graph_id(node_id, False)]: info_is_hidden = False

        # set node color (clicked/failed)
        info_color = DEFAULT_COLOR
        if to_graph_id(node_id, False) == clicked_node: # node was clicked on
            info_color = CLICKED_COLOR
        elif to_graph_id(info_id, True) == clicked_node: # info was clicked on
            info_color = CLICKED_COLOR
        elif clicked_node in G.pred[to_graph_id(node_id, False)]:
            info_color = NEIGHBOR_COLOR
        # color failed nodes (overwrites clicked coloring if relevant)
        if (info_data[info_data["id"] == info_id]["failed"].values[0] == "t"):
            info_color = FAILED_COLOR

        vert_id = node_data[node_data["id"] == node_id]["vertex_id"].values[0]

        G.add_node(
            to_graph_id(info_id, is_info),
            color=info_color,
            degree=degree,
            is_info=is_info,
            label=create_label(info_id),
            labelHighlightBold=True,
            origin_id=node_id,
            shape=DEFAULT_INFO_SHAPE,
            vertex_id=vert_id,
            hidden=info_is_hidden
        )
        G.add_edge(to_graph_id(node_id, False), to_graph_id(info_id, is_info))

def add_edges_to_networkx(G, degree, trial_maker_id, node_id):
    """ Add incoming edges of the given node to G, using dependent_vertex_ids.
    """
    node_data = node_data_by_trial_maker[trial_maker_id]
    node_id = from_graph_id(node_id)[0]
    deg_nodes = node_data[node_data["degree"] == degree]

    # get dependent vertices (incoming edges) in a list
    dependent_vertices = node_data[node_data["id"] == node_id]["dependent_vertex_ids"].values[0].strip('][').split(',')

    # find the corresponding row of deg_nodes for each vertex_id
    dependent_nodes = [deg_nodes[deg_nodes["vertex_id"] == float(v)] for v in dependent_vertices]

    # extract the node_id from each dependent vertex row
    dependent_nodes = [n["id"].values[0] for n in dependent_nodes]

    # add as edges (dependent node --> curent node)
    is_info = False
    edge_list = [(to_graph_id(int(dependent_node), is_info), to_graph_id(int(node_id), is_info)) for dependent_node in dependent_nodes]
    G.add_edges_from(edge_list)

def generate_graph(graph_settings):
    ''' Given a dictionary of graph settings, containing: degree (int or float), trial_maker_id (string), whether to show infos (bool), whether to show outgoing nodes (bool) and/or incoming nodes (bool), which solver to use, a seed, and the id of the clicked node,  return a DiGraph containing the nodes (with metadata from infos) and edges in that degree and associated with that trial_maker_id. Some nodes or infos will have the "hidden" attribute set to True depending on the settings, but all of them will be in the DiGraph.

    show_outgoing: shows only nodes with outgoing edges from the clicked node
    show_incoming: shows only nodes with incoming edges from clicked node
    if both are true, then all nodes connected to the clicked node are shown
    if both are false, all nodes are shown
    '''
    # validation: ensure settings dictionary is correctly passed in
    for setting in GRAPH_SETTINGS:
        if setting not in graph_settings:
            raise Exception("Invalid graph settings")

    # validation: ensure trial_maker_id is valid
    if graph_settings[EXP] not in node_data_by_trial_maker.keys():
        raise Exception("Invalid trial_maker_id.")

    # validation: ensure clicked_node is present in the data
    clicked_graph_id, clicked_is_info = from_graph_id(graph_settings[CLICKED_NODE])
    if clicked_is_info:
        if clicked_graph_id not in info_data_by_trial_maker[graph_settings[EXP]]["id"].values:
            raise ClickedNodeException
    elif clicked_is_info != None:
        if clicked_graph_id not in node_data_by_trial_maker[graph_settings[EXP]]["id"].values:
            raise ClickedNodeException

    # use correct data for that trial_maker_id
    node_data = node_data_by_trial_maker[graph_settings[EXP]]

    # create graph
    G = nx.DiGraph()

    # add nodes+edges from node_data to the Graph, and associated infos
    deg_nodes = node_data[node_data["degree"] == graph_settings[DEGREE]]
    for node_id in deg_nodes["id"].values.tolist():
        add_node_to_networkx(G, graph_settings[DEGREE], graph_settings[EXP], node_id, graph_settings[CLICKED_NODE], graph_settings[SHOW_OUTGOING], graph_settings[SHOW_INCOMING])
        add_edges_to_networkx(G, graph_settings[DEGREE], graph_settings[EXP], node_id)
        add_infos_to_networkx(G, graph_settings[DEGREE], graph_settings[EXP], node_id, graph_settings[CLICKED_NODE], graph_settings[SHOW_INFOS])

    return G

#----------------------------- Routes --------------------------------
@app.route('/setclickednode', methods=['GET'])
def set_clicked_node():
    """ Sets clicked_node cookie.
    """
    response = make_response('')

    clicked_node = request.args.get('id')
    if from_graph_id(clicked_node)[1] is None:
        raise Exception("Error in graph id to set.")

    response.set_cookie(CLICKED_NODE, clicked_node)

    return response

@app.route('/updateclickednode', methods=['GET'])
def update_clicked_node_route():
    """ Given a graph_id, degree, and trial_maker_id, finds the vertex
    associated with the graph_id and returns that vertex in the given degree
    and trial_maker_id, or "" if not found.
    """
    graph_id = request.args.get('graph_id')
    degree = request.args.get('degree')
    trial_maker_id = request.args.get('trial_maker_id')
    # if any are still None, the args are invalid, so return ""
    if graph_id is None or degree is None or trial_maker_id is None:
        return make_response("")

    new_node_id = update_clicked_node(graph_id, degree, trial_maker_id)
    return make_response(new_node_id)

@app.route('/getcontent', methods=['GET'])
def get_content():
    # get arguments out of request
    exp = request.args.get('exp')
    id = request.args.get(CLICKED_NODE)
    if id is None:
        id = request.cookies.get(CLICKED_NODE)

    # get content list and convert to html
    content_list = get_content_list(exp, id)
    content_html = ''

    for content_string in content_list:
        content_html += content_string + '<br>'

    return make_response(content_html)


@app.route('/getgraph', methods=['GET'])
def get_graph(from_index=False):
    # process data into dicts (global variables)
    data_path = app.config.get('data_path')
    if data_path[-1] != "/":
        data_path += "/"
    process_data(data_path)

    # get the settings
    settings = get_settings(request, from_index=from_index)
    min_degree = node_data_by_trial_maker[settings[EXP]]["degree"].min()

    # create network layout, based on layout generated on minimal degree
    global vertex_pos
    if (vertex_pos is None) or (settings[LAYOUT] != request.cookies.get(LAYOUT)) or (settings[SEED] is not None):
        # print("(Re)setting global position")
        vertex_pos = {}

        # get graph settings for minimal degree
        pos_settings = get_settings(request, from_index=from_index)
        pos_settings[DEGREE] = min_degree

        try:
            pos_graph = generate_graph(pos_settings)
        except ClickedNodeException:
            pos_settings[CLICKED_NODE] = ''
            pos_graph = generate_graph(pos_settings)

        if LAYOUT_OPTIONS[pos_settings[LAYOUT]]['has_seed']:
            pos = LAYOUT_OPTIONS[pos_settings[LAYOUT]]['func'](pos_graph, seed=pos_settings[SEED])
        else:
            pos = LAYOUT_OPTIONS[pos_settings[LAYOUT]]['func'](pos_graph)

        # convert to vertex-id-mapped position dict, with only node positions added
        vertex_id_map = pos_graph.nodes(data='vertex_id')
        for graph_id, xy in pos.items():
            v_id = int(vertex_id_map[graph_id])
            if graph_id[0] == 'n':
                vertex_pos[v_id] = {'x': xy[0] , 'y': xy[1]}

    # create network
    pyvis_net = Network(directed=True)
    try:
        nx_graph = generate_graph(settings)
    except ClickedNodeException:
        settings[CLICKED_NODE] = ''
        nx_graph = generate_graph(settings)

    # use the correct solver (barnes hut is the default)
    if settings[SOLVER] == FORCE_ATLAS_2BASED:
        pyvis_net.force_atlas_2based()
    elif settings[SOLVER] == REPULSION:
        pyvis_net.repulsion()
    elif settings[SOLVER] == HIERARCHICAL_REPULSION:
        pyvis_net.hrepulsion()

    # read networkx graph into pyvis, add necessary attributes
    pyvis_net.from_nx(nx_graph)
    for (graph_id, node) in pyvis_net.node_map.items():
        # position
        v_id = node['vertex_id']
        node['x'] = vertex_pos[v_id]['x'] * LAYOUT_OPTIONS[settings[LAYOUT]]['scale'] # scaling necessary for x,y position to work
        node['y'] = vertex_pos[v_id]['y'] * LAYOUT_OPTIONS[settings[LAYOUT]]['scale']

        # overwrite incorrect neighbor color (leftover from prev degree)
        if node['color'] == NEIGHBOR_COLOR and settings[CLICKED_NODE] not in nx_graph.nodes:
            node['color'] = DEFAULT_COLOR

    # generate default html with pyvis template
    graph_html = pyvis_net.generate_html()

    # modify pyvis html script
    script_to_replace = 'network = new vis.Network(container, data, options);'
    click_script = 'network = new vis.Network(container, data, options);\
        network.on("click", function(properties) {\
            let node_id = properties.nodes[0];\
            if (node_id != undefined) {\
                node_form_input = document.getElementById("clicked-node-input");\
                node_form_input.value = node_id;\
                getContent(from_click=true);\
                getGraph(from_click=true);\
            }\
        })'

    graph_html = graph_html.replace(script_to_replace, click_script)

    # remove parts of HTML document
    html_strings_to_remove = [
        '<html>\n',
        '</html>',
        '<head>\n',
        '</head>\n',
        '<body>\n',
        '</body>',
        '<meta charset="utf-8">\n'
    ]
    for html_string in html_strings_to_remove:
        graph_html = graph_html.replace(html_string, '')

    if from_index:
        return graph_html

    response = make_response(graph_html)
    set_graph_cookies(response, settings)

    return response

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def create_visualizer():
    # generate the graph HTML
    graph_html = get_graph(from_index=True)

    # get the settings
    settings = get_settings(request, from_index=True)

    # create values to fill in for page template
    node_data = node_data_by_trial_maker[settings[EXP]]
    min_degree = node_data["degree"].min()
    max_degree = node_data["degree"].max()
    min_vertex_id = node_data["vertex_id"].min()
    max_vertex_id = node_data["vertex_id"].max()

    trialmaker_options = node_data_by_trial_maker.keys()

    show_nodes_val = SHOW_NODES_ALL
    if settings[SHOW_OUTGOING] and settings[SHOW_INCOMING]:
        show_nodes_val = SHOW_NODES_CONNECTED
    elif settings[SHOW_OUTGOING]:
        show_nodes_val = SHOW_NODES_OUTGOING
    elif settings[SHOW_INCOMING]:
        show_nodes_val = SHOW_NODES_INCOMING

    # render template and make response
    page_html = render_template(
        'dashboard_visualizer.html',
        graph=graph_html,
        trialmaker_options=trialmaker_options,
        degree_min=min_degree,
        degree_max=max_degree,
        degree_placeholder=int(settings[DEGREE]),
        find_min=min_vertex_id,
        find_max=max_vertex_id,
        content=get_content_list(settings[EXP], settings[CLICKED_NODE]),
        show_infos_checked=("checked" if settings[SHOW_INFOS] else ""),
        show_nodes_option = show_nodes_val,
        layout_options = LAYOUT_OPTIONS,
        selected_layout = settings[LAYOUT],
        solver_options = VALID_SOLVERS,
        selected_solver = settings[SOLVER],
        )
    response = make_response(page_html)

    set_graph_cookies(response, settings)

    return response