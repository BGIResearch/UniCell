#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ontoGRAPH.py
# @Time      :2024/3/31 21:27
# @Author    :Luni Hu

import os

import numpy as np
import obonet
import networkx as nx
from collections import Counter, OrderedDict
import torch
from tqdm import tqdm


class OntoGRAPH(object):
    def __init__(self, graph = None, cell_type=None):
        if graph is not None:
            self.graph = graph
            print("Graph loaded")
        else:
            self.graph = self.import_ontology()
        nodes = self.graph.nodes
        nodes = [node for node in nodes if "CL:" in node]
        self.graph = self.graph.subgraph(nodes)
        self.obsolete_dict = self.get_obsolete_id_mapper()

        if cell_type is not None:
            
            cell_type = [self.obsolete_dict.get(c, c) for c in cell_type]
            cell_type = [c for c in cell_type if c in self.graph.nodes]

            # Find the ancestors of each node
            ancestors = [set(nx.ancestors(self.graph, node)) if len(set(nx.ancestors(self.graph, node))) > 0 else set([node]) for node in cell_type]

            # Find the lowest common ancestor
            common_ancestors = set.intersection(*ancestors)
            print(common_ancestors)

            lca = min(common_ancestors, key=lambda node: min([nx.shortest_path_length(self.graph, node, n) for n in cell_type]))
            print("Lowest Common Ancestor:", lca)
            self.common_ancestor = lca

            nodes = self.get_all_descendants(nodes=lca, inclusive=True)
            self.graph = self.graph.subgraph(nodes)

            ancestors = [set(nx.ancestors(self.graph, node)) for node in cell_type]
            all_ancestors = set.union(*ancestors)
            all_nodes = set(all_ancestors).union(set(cell_type))

            self.graph = self.graph.subgraph(all_nodes)
            graph = self.graph.subgraph([lca]+cell_type)

            all_nodes = set(graph.nodes)

            for node in tqdm(cell_type):

                if not nx.has_path(graph, lca, node):

                    all_paths = list(nx.all_simple_paths(self.graph, lca, node))

                    path_keep = min(all_paths, key=len)

                    all_nodes = all_nodes.union(set(path_keep))

            self.graph = self.graph.subgraph(all_nodes)

            print("get id mapper ...")
            self.id2name = self.get_id_mapper()
            print("convert to vocab ...")
            self.vocab = self.convert_to_vocab()
            print("get hierarchical levels ...")
            self.hierarchical_levels = self.get_hierarchical_levels()
            print("get hierarchical array...")
            self.hierarchical_array = self.get_hierarchical_array()
            print("get cell type idx set ...")
            self.cell_type_idx_set = self.get_cell_type_idx_set()

    def import_ontology(self, url=None):
        """Import taxrank cell ontology.

        Parameters
        ----------
        url: str, default: "/gstore/data/omni/scdb/cell-ontology-2022-09-15/cl-basic.obo"
            The url of the ontology obo file.

        Returns
        -------
        dgl.DGLGraph
            Node graph of ontology.
        """
        if url is None:
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            url = os.path.join(curr_dir, "cl-basic.obo")

        graph = obonet.read_obo(url, ignore_obsolete=False).reverse()  # flip for intuitiveness

        dgl_graph = nx.MultiDiGraph((u, v, k, data) for u, v, k, data in graph.edges(keys=True, data=True)
                                    if k in ["is_a"]) # no "develops_from" relation

        dgl_graph.add_nodes_from((n, data) for n, data in graph.nodes(data=True))

        return dgl_graph  # return as DGL graph

    def get_obsolete_id_mapper(self):

        obsolete_dict = {id_: data["replaced_by"][0] for id_, data in self.graph.nodes(data=True)
                         if "replaced_by" in data.keys()}

        obsolete_dict = {k:v for k, v in obsolete_dict.items() if v in self.graph.nodes}

        return obsolete_dict
    def get_id_mapper(self) -> dict:
        """Mapping from term ID to name.

        Returns
        -------
        dict
            The id to name mapping dictionary.
        """
        return {id_: data.get("name") for id_, data in self.graph.nodes(data=True)}

    def convert_to_vocab(self):
        """Convert the cell type IDs to a vocabulary dictionary.

        Returns
        -------
        dict
            A dictionary mapping cell type IDs to their corresponding indices in the vocabulary.
        """
        ids = sorted(self.id2name.keys())
        return OrderedDict({id: idx for idx, id in enumerate(ids)})

    def get_siblings(self, term_id):
        parent_id = list(self.graph.predecessors(term_id))[0]  # Assuming only one parent
        siblings = list(self.graph.successors(parent_id))
        siblings.remove(term_id)  # Remove the term itself from the list of siblings
        return siblings

    def get_parents(self, term_id):
        parents = list(self.graph.predecessors(term_id))
        return parents

    def get_children(self, term_id):
        children = list(self.graph.successors(term_id))
        return children

    def get_all_ancestors(self, node, inclusive=False) -> nx.DiGraph:
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

        """

        ancestors = nx.ancestors(self.graph, node)

        if inclusive:
            ancestors = ancestors | {node}

        return list(ancestors)

    def get_hierarchical_levels(self):
        """Get hierarchical labels for each node in the ontology.

        Returns
        -------
        dict
            A dictionary mapping each node ID to its hierarchical labels.
        """
        hierarchical_labels = OrderedDict()

        # Find the root node by identifying the node with no incoming edges
        roots = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]
        if len(roots) == 1:
            root = roots[0]
        else:
            raise ValueError("Unable to determine the root node of the ontology.")

        def assign_labels(node_id, labels_dict):
            if node_id is root:
                labels_dict[node_id] = 0
            else:
                all_paths = nx.all_simple_paths(self.graph, root, node_id)
                path_length = max([len(path) for path in all_paths])-1

                labels_dict[node_id] = path_length

        # Assign hierarchical labels to each node
        for node_id in self.graph.nodes:
            assign_labels(node_id, hierarchical_labels)

        return hierarchical_labels

    def get_all_descendants(self, nodes, inclusive=True):
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

        """

        if isinstance(nodes, str):  # one term id
            descendants = nx.descendants(self.graph, nodes)

        if inclusive:
            descendants = descendants | {nodes}

        return list(descendants)

    def get_hierarchical_array(self):

        num_level = max(list(self.hierarchical_levels.values())) + 1
        num_classes_per_level = Counter(list(self.hierarchical_levels.values()))
        num_nodes = self.graph.number_of_nodes()
        node_hclasses = self.get_node_hclasses()
        harray = [torch.zeros(num_nodes, num_classes_per_level[idx]) for idx in range(num_level)]


        for k, v in self.hierarchical_levels.items():

            for level in range(v):

                ancestors = self.get_all_ancestors(k)

                ancestors = ancestors
                ancestors_in_this_level = [ancestor for ancestor in ancestors if self.hierarchical_levels[ancestor] == level]

                for ancestor in ancestors_in_this_level:

                    ancestor_idx = node_hclasses[level][ancestor]

                    harray[level][self.vocab[k], ancestor_idx] = 1

            harray[v][self.vocab[k], node_hclasses[v][k]] = 1


        return harray

    def get_node_hclasses(self):

        num_classes_per_levels = Counter(list(self.hierarchical_levels.values()))
        max_levels = max(num_classes_per_levels.keys())

        node_hclasses = {}

        for level in range(max_levels+1):
            nodes_in_this_level = [k for k, v in self.hierarchical_levels.items() if v == level]
            nodes_class_in_this_level = {node: idx for idx, node in enumerate(nodes_in_this_level)}
            node_hclasses[level] = nodes_class_in_this_level

        return node_hclasses

    def get_cell_type_idx_set(self):
        """
        {
            global_id: {
                local_id : vocab_id
            }
        }
        """

        cell_type_idx_set = {}
        cell_type_hierarchy_level = self.hierarchical_levels
        hierarchical_array = self.hierarchical_array
        cell_type_vocab = self.vocab
        for k, v in cell_type_hierarchy_level.items():
            cell_type = cell_type_vocab[k]
            level_class = np.nonzero(hierarchical_array[v][cell_type, :])[0]
            level_class = level_class.item()
            if v not in cell_type_idx_set:
                cell_type_idx_set.update({v: {level_class: cell_type}})
            else:
                cell_type_idx_set[v].update({level_class: cell_type})

        return cell_type_idx_set

    def pickle(self, output):
        import pickle
        nx.write_gml(self.graph, os.path.join(output, 'ontoGraph.graph.gml'))
        self.graph = None
        with open(os.path.join(output, 'ontoGraph.pk'), 'wb') as w:
            pickle.dump(self, w)
