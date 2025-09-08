#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ontologies.py
# @Time      :2024/3/26 17:07
# @Author    :Luni Hu

import networkx as nx
import obonet
from typing import Union, Tuple, List

import os
curr_dir = os.path.dirname(__file__)


def subset_nodes_to_set(nodes, restricted_set: Union[list, set]) -> nx.DiGraph:
    """Restrict nodes to a given set.

    Parameters
    ----------
    nodes: networkx.DiGraph
        Node graph.
    restricted_set: list, set
        Restricted node list.

    Returns
    -------
    networkx.DiGraph
        Node graph of restricted set.

    Examples
    --------
    >>> subset_nodes_to_set(nodes, node_list)
    """

    return {node for node in nodes if node in restricted_set}


def import_ontology(
    url = None,
) -> nx.DiGraph:
    """Import taxrank cell ontology.

    Parameters
    ----------
    url: str, default: "/gstore/data/omni/scdb/cell-ontology-2022-09-15/cl-basic.obo"
        The url of the ontology obo file.

    Returns
    -------
    networkx.DiGraph
        Node graph of ontology.

    Examples
    --------
    >>> onto = import_ontology()
    """
    if url is None:
        url = os.path.join(curr_dir, "cl-basic.obo")

    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph

def get_id_mapper(graph) -> dict:
    """Mapping from term ID to name.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.

    Returns
    -------
    dict
        The id to name mapping dictionary.

    Examples
    --------
    >>> id2name = get_id_mapper(onto)
    """

    return {id_: data.get("name") for id_, data in graph.nodes(data=True)}
def get_all_descendants(graph, nodes, node_list=None, inclusive=False) -> nx.DiGraph:
    """Get all descendant nodes of given node(s).

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    nodes: str, list
        ID of given node or a list of node IDs.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.
    inclusive: bool, default: False
        Whether to include the given node in the results.

    Returns
    -------
    networkx.DiGraph
        Node graph of descendants.

    Examples
    --------
    >>> descendants = get_all_descendants(onto, id)
    """

    if isinstance(nodes, str):  # one term id
        descendants = nx.descendants(graph, nodes)
    else:  # list of term ids
        descendants = set.union(*[nx.descendants(graph, node) for node in nodes])

    if inclusive:
        descendants = descendants | {nodes}

    if node_list is None:
        return descendants
    return subset_nodes_to_set(descendants, node_list)


def get_all_ancestors(graph, node, node_list=None, inclusive=False) -> nx.DiGraph:
    """Get all ancestor nodes of a given node.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.
    inclusive: bool, default: False
        Whether to include the given node in the results.

    Returns
    -------
    networkx.DiGraph
        Node graph of ancestors.

    Examples
    --------
    >>> ancestors = get_all_ancestors(onto, id)
    """

    ancestors = nx.ancestors(graph, node)
    if inclusive:
        ancestors = ancestors | {node}

    if node_list is None:
        return ancestors
    return subset_nodes_to_set(ancestors, node_list)
