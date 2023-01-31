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
        if graph_id in ['undefined', None]:
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
        return ["An error has occurred. Content cannot be displayed."]
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
def process_data():
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
    nodes = pd.read_csv(PATH + "node.csv", low_memory=False)
    networks = pd.read_csv(PATH + "network.csv", low_memory=False)
    infos = pd.read_csv(PATH + "info.csv")

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

        # filter infos like nodes, sort
        info_data = infos
        info_data = info_data[infos["type"] == "graph_chain_trial"]
        info_data = info_data[["id", "creation_time", "details", "origin_id", "network_id", "participant_id", "failed"]] # TODO: Probably want more columns here
        info_data = info_data.sort_values(["network_id", "origin_id"])

        # add filtered Dataframes to the global dicts
        node_data_by_trial_maker[trial_maker_id] = node_data
        info_data_by_trial_maker[trial_maker_id] = info_data

    processing_done = True

def get_settings(request, from_index=False):
    ''' Get settings from the request, or set correct defaults.
        Must be run after process_data() so that the trial_maker_id validation works properly.
        Argument: request
        Returns: tuple of:
            clicked_node (graph ID)
            exp (string)
            degree (float)
            show_infos (bool)
    '''
    # check that data has been processed
    if len(list(node_data_by_trial_maker.keys())) == 0:
        raise Exception("Settings cannot be found before data is processed.")

    # get clicked node id
    clicked_node = request.args.get('clicked-node')
    if clicked_node is None:
        clicked_node = request.cookies.get('clicked-node')
    clicked_node = clicked_node if (clicked_node is not None) else ''

    # find the correct 'exp' (trial maker id)
    exp = request.args.get('trial-maker-id')
    if exp is None or exp not in node_data_by_trial_maker.keys():
        exp = list(node_data_by_trial_maker.keys())[0]

    # find the correct 'degree'
    degree = request.args.get('degree')
    if degree in [None, '']:
        degree_cookie = request.cookies.get('degree')
        if degree_cookie is None:
            degree = node_data_by_trial_maker[exp]["degree"].min()
        else:
            degree = float(degree_cookie)

    # check whether show infos is on, convert to boolean
    if from_index:
        show_infos = request.cookies.get('show-infos')
    else:
        show_infos = request.args.get('show-infos')
    show_infos = True if (show_infos == "true") else False

    return (clicked_node, exp, degree, show_infos)

def add_node_to_networkx(G, degree, trial_maker_id, node_id, clicked_node):
    """ Adds node to networkx DiGraph.
        Arguments:
            G: networkx DiGraph
            degree: float
            trial_maker_id: string
            node_id: int
            clicked_node: string (in graph_id format)
        Node attributes:
            graph_id    ('n' + 'id' field)
            color
            degree
            is_info     (False)
            label       ('id' with spacing)
            labelHighlightBold (True)
            shape       (DEFAULT_NODE_SHAPE)
            vertex_id   ('vertex_id' field)
    """
    # filter data
    node_data = node_data_by_trial_maker[trial_maker_id]
    info_data = info_data_by_trial_maker[trial_maker_id]
    clicked_graph_id, clicked_is_info = from_graph_id(clicked_node)

    # extract vertex id
    vert_id = node_data[node_data["id"] == node_id]["vertex_id"].values[0]

    # set node color (clicked/failed)
    node_color = DEFAULT_COLOR
    if to_graph_id(node_id, False) == clicked_node: # node was clicked on
            node_color = CLICKED_COLOR
    elif clicked_is_info and info_data[info_data["id"] == clicked_graph_id]["origin_id"].values[0] == node_id: # node's info was clicked on
        node_color = CLICKED_COLOR
    # color failed nodes (overwrites clicked coloring if relevant)
    if (node_data[node_data["id"] == node_id]["failed"].values[0] == "t"):
        node_color = FAILED_COLOR

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

def add_infos_to_networkx(G, degree, trial_maker_id, node_id, clicked_node, show_infos):
    """ Adds infos associated with given node_id to networkx DiGraph, and edges.
        Arguments:
            G: networkx DiGraph
            degree: float
            trial_maker_id: string
            node_id: int
            clicked_node: string (in graph_id format)
            show_infos: bool
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

        info_is_hidden = not (show_infos and ((to_graph_id(node_id, False) == clicked_node) or (to_graph_id(info_id, True) == clicked_node)))

        # set node color (clicked/failed)
        info_color = DEFAULT_COLOR
        if to_graph_id(node_id, False) == clicked_node: # node was clicked on
            info_color = CLICKED_COLOR
        elif to_graph_id(info_id, True) == clicked_node: # info was clicked on
            info_color = CLICKED_COLOR
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

def add_edges_to_networkx(G, degree, trial_maker_id, row):
    """ Add incoming edges of the given node to G, using dependent_vertex_ids.
    """
    node_data = node_data_by_trial_maker[trial_maker_id]
    node_id = row["id"]
    deg_nodes = node_data[node_data["degree"] == degree]

    # get dependent vertices (incoming edges) in a list
    dependent_vertices = row["dependent_vertex_ids"].strip('][').split(',')

    # find the corresponding row of deg_nodes for each vertex_id
    dependent_nodes = [deg_nodes[deg_nodes["vertex_id"] == float(v)] for v in dependent_vertices]

    # extract the node_id from each dependent vertex row
    dependent_nodes = [n["id"].values[0] for n in dependent_nodes]

    # add as edges (dependent node --> curent node)
    is_info = False
    edge_list = [(to_graph_id(int(dependent_node), is_info), to_graph_id(int(node_id), is_info)) for dependent_node in dependent_nodes]
    G.add_edges_from(edge_list)

def generate_graph(degree, trial_maker_id, show_infos, clicked_node):
    ''' Given a degree (int or float) and a trial_maker_id in the experiment,
    whether to show infos (bool), and the id of the clicked node, return a DiGraph containing the nodes (with metadata from infos) and edges in that degree and associated with that trial_maker_id.
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

    # validation: ensure clicked_node is present in the data
    clicked_graph_id, clicked_is_info = from_graph_id(clicked_node)
    if clicked_is_info:
        if clicked_graph_id not in info_data_by_trial_maker[trial_maker_id]["id"].values:
            raise Exception("Error: Clicked info is not present in data.")
    elif clicked_is_info != None:
        if clicked_graph_id not in node_data_by_trial_maker[trial_maker_id]["id"].values:
            raise Exception("Error: Clicked node is not present in data.")

    # use correct data for that trial_maker_id
    node_data = node_data_by_trial_maker[trial_maker_id]

    # create graph
    G = nx.DiGraph()

    # add nodes from node_data to the Graph, and associated infos
    deg_nodes = node_data[node_data["degree"] == degree]
    for node_id in deg_nodes["id"].values.tolist():
        add_node_to_networkx(G, degree, trial_maker_id, node_id, clicked_node)
        add_infos_to_networkx(G, degree, trial_maker_id, node_id, clicked_node, show_infos)

    # add edges to the Graph: iterate over deg_nodes, add incoming edges
    # using dependent_vertex_ids column
    for _, ser in deg_nodes.iterrows():
        add_edges_to_networkx(G, degree, trial_maker_id, ser)

    return G

#----------------------------- Routes --------------------------------
@app.route('/setclickednode', methods=['GET'])
def set_clicked_node():
    response = make_response('')

    clicked_node = request.args.get('id')
    if from_graph_id(clicked_node)[1] is None:
        raise Exception("Error in graph id to set.")

    response.set_cookie('clicked-node', clicked_node)

    return response

@app.route('/getcontent', methods=['GET'])
def get_content():
    # get arguments out of request
    exp = request.args.get('exp')
    id = request.args.get('clicked-node')
    if id is None:
        id = request.cookies.get('clicked-node')

    # get content list and convert to html
    content_list = get_content_list(exp, id)
    content_html = ''

    for content_string in content_list:
        content_html += content_string + '<br>'

    return make_response(content_html)


@app.route('/getgraph', methods=['GET'])
def get_graph(from_index=False):
    # process data into dicts (global variables)
    process_data()

    # get the settings
    clicked_node, exp, degree, show_infos = get_settings(request, from_index=from_index)

    # create network
    pyvis_net = Network(directed=True)
    nx_graph = generate_graph(degree, exp, show_infos, clicked_node)

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
                \
                getContent();\
                getGraph();\
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

    response.set_cookie('degree', str(degree))
    response.set_cookie('exp', exp)
    response.set_cookie('show-infos', "true" if show_infos else "false")
    response.set_cookie('clicked-node', clicked_node)

    return response

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def create_visualizer():
    # generate the graph HTML
    graph_html = get_graph(from_index=True)

    # get the settings
    clicked_node, exp, degree, show_infos = get_settings(request, from_index=True)

    # create values to fill in for page template
    node_data = node_data_by_trial_maker[exp]
    min_degree = node_data["degree"].min()
    max_degree = node_data["degree"].max()
    min_vertex_id = node_data["vertex_id"].min()
    max_vertex_id = node_data["vertex_id"].max()

    trialmaker_options = node_data_by_trial_maker.keys()

    # render template and make response
    page_html = render_template(
        'dashboard_visualizer.html',
        graph=graph_html,
        trialmaker_options=trialmaker_options,
        degree_min=min_degree,
        degree_max=max_degree,
        degree_placeholder=int(degree),
        physics_options=["barnes hut", "placeholder 1"],
        find_min=min_vertex_id,
        find_max=max_vertex_id,
        content=get_content_list(exp, clicked_node),
        show_infos_checked=("checked" if show_infos else "")
        )
    response = make_response(page_html)

    # set cookies and return response
    response.set_cookie('degree', str(degree))
    response.set_cookie('exp', exp)
    response.set_cookie('show-infos', ("true" if show_infos else "false"))
    response.set_cookie('clicked-node', clicked_node)

    return response