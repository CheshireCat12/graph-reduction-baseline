import os
import pathlib
import xml.dom.minidom as md
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import torch_geometric.utils as tutils
from torch_geometric.datasets import TUDataset
from tqdm import tqdm


############################################
#                  Loader                  #
############################################

def load_graphs_from_TUDataset(root: str,
                               name_dataset: str) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Use the Pytorch Geometric (PyG) loader to download the graphs from the TUDataset.
    The raw graphs from PyG are saved in `root`.

    The created NetworkX graphs have a node attribute `x` that is an `np.array`.
    The corresponding class of each graph is also retrieved from the TUDataset graphs.

    Args:
        root: Path where to save the raw graph dataset
        name_dataset: Name of the graph dataset to load

    Returns:
        List of the loaded NetworkX graphs and `np.array` of the corresponding class of each graph
    """
    dataset = TUDataset(root=root, name=name_dataset)

    node_attr = 'x'

    # Convert the PyG graphs into NetworkX graphs
    nx_graphs = [tutils.to_networkx(graph, node_attrs=[node_attr], to_undirected=True)
                 for graph in dataset]

    # Cast the node attribute x from list into np.array
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            nx_graph.nodes[idx_node][node_attr] = np.array(data_node[node_attr])

    graph_cls = np.array([int(graph.y) for graph in dataset])

    return nx_graphs, graph_cls


############################################
#              graph writer                #
############################################

def _modify_node_type_to_str(graph: nx.Graph) -> nx.Graph:
    """
    Modify the type of the node attribute.
    A shallow copy of the graph is created to modify the node attribute's type
    Change the `np.ndarray` or `list` node attribute `x` into `str` attribute

    Args:
        graph: Graph to modify the type of the nodes' attribute

    Returns:
        Modified copy of the graph
    """
    # TODO: Handle the np.ndarray node attribute
    # TODO: Should the attr_node 'x' be a function parameter?
    node_attr = 'x'
    new_graph = graph.copy()

    for idx_node, data_node in graph.nodes(data=True):
        new_graph.nodes[idx_node][node_attr] = str(data_node[node_attr])

    return new_graph


def _write_classes(graph_cls: np.ndarray,
                   filename: str) -> None:
    """
    Save the class of each graph in a tuple (graph_name, graph_cls).

    Args:
        graph_cls: List of graph classes. The idx in the array of
                   the class must correspond to the graph idx to which it belongs.
        filename: Filename where to save the graph classes

    Returns:

    """
    graph_collection = ET.Element('GraphCollection')

    idx_graph_to_classes = ET.SubElement(graph_collection, 'idx_graph_to_classes')

    for idx_graph, cls in enumerate(graph_cls):
        element = ET.SubElement(idx_graph_to_classes, 'element')
        element.set('graph_file', f'gr_{idx_graph}.graphml')
        element.set('class', str(cls))

    b_xml = ET.tostring(graph_collection).decode()
    newxml = md.parseString(b_xml)

    with open(filename, mode='w') as f:
        f.write(newxml.toprettyxml(indent=' ', newl='\n'))


def save_graphs(path: str,
                graphs: List[nx.Graph],
                graph_cls: Optional[np.ndarray] = None) -> None:
    """
    Save the given graphs in `path` under `.graphml` format.
    The saved graphs are named according to their position in the list
    (e.g., the first graph in the list is named `graph_0.graphml`).

    The `np.ndarray` node attribute `x` is modified into `str` attribute

    Args:
        path: Path to the folder where to save the graphs.
            If the folder doesn't exist it is created.
        graphs: List of graphs to save
        graph_cls:
    """
    # Make sure that the path to the folder exist, if not create it.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for idx_graph, graph in tqdm(enumerate(graphs),
                                 total=len(graphs),
                                 desc='Save Graphs'):
        # Change the np.ndarray or list node attribute to str (graph copy)
        copied_graph = _modify_node_type_to_str(graph)

        filename = f'graph_{idx_graph}.graphml'
        path_to_graph = os.path.join(path, filename)
        nx.write_graphml_lxml(copied_graph, path_to_graph, prettyprint=True)

    if graph_cls is not None:
        filename_cls = os.path.join(path, 'graph_classes.cxl')
        _write_classes(graph_cls, filename_cls)
