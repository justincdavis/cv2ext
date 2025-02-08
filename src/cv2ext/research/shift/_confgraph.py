# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


def _clamp(n: float, _min: float, _max: float) -> float:
    if n < _min:
        return _min
    if n > _max:
        return _max
    return n


def create_graph(
    all_nodes: set[str],
    all_edges: dict[str, set],
    all_weights: dict[str, dict[str, float]],
    min_connects: float = 1,
    cutoff: float = 0,
    upper_outlier_cutoff: float = 80,
    graph_type: type[nx.Graph | nx.DiGraph] = nx.Graph,
) -> nx.Graph | nx.DiGraph:
    """
    Create a networkx graph from the edges and weights.

    Parameters
    ----------
    all_nodes : set[str]
        The nodes to add to the graph.
    all_edges : dict[str, set]
        The edges to add to the graph.
    all_weights : dict[str, dict[str: float]]
        The weights to add to the graph.
    min_connects : float, optional
        The minimum number of connections an edge must have to not delete it
        By default 1.
    cutoff : float, optional
        The cutoff for the edge weights.
        By default 0.
    upper_outlier_cutoff : float, optional
        The cutoff for the upper outlier percentage.
        Any values in the top (100 - cutoff)% percent will be clamped.
        Helps to remove takeover edges.
        By default 80.
    graph_type : type[nx.Graph | nx.DiGraph], optional
        The type of networkx graph to use.
        By default nx.Graph.
        Options are: [nx.Graph, nx.DiGraph].

    Returns
    -------
    nx.Graph | nx.DiGraph
        The graph created from the edges and weights.

    """
    new_edges = defaultdict(set)
    new_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # get max_edge_weight
    max_edge_weight = -1.0
    for primary_node, secondary_nodes in all_edges.items():
        for secondary_node in secondary_nodes:
            max_edge_weight = max(
                max_edge_weight,
                all_weights[primary_node][secondary_node],
            )

    # check if edge meets minimum number of connections
    # clamp edges in the top (100 - cutoff)% percent
    # trim edges below the cutoff
    for primary_node, secondary_nodes in all_edges.items():
        local_weights = [
            all_weights[primary_node][secondary_node]
            for secondary_node in secondary_nodes
        ]
        upper_limit = float(np.percentile(local_weights, upper_outlier_cutoff))
        for secondary_node in secondary_nodes:
            current_weight = all_weights[primary_node][secondary_node]
            if current_weight <= min_connects:
                continue
            current_weight = _clamp(current_weight, 1, upper_limit)
            new_weight = 1.0 - all_weights[primary_node][secondary_node] / upper_limit
            if new_weight >= cutoff:
                new_edges[primary_node].add(secondary_node)
                new_weights[primary_node][secondary_node] = new_weight

    # create the graph
    graph = graph_type()
    for node in all_nodes:
        graph.add_node(node)
    for primary_node, secondary_nodes in new_edges.items():
        for secondary_node in secondary_nodes:
            graph.add_edge(
                primary_node,
                secondary_node,
                weight=new_weights[primary_node][secondary_node],
            )
    return graph


def get_node(modelname: str, conf_val: float, num_bins: int) -> str:
    """
    Get the string represnetation of a node.

    Parameters
    ----------
    modelname : str
        The name of the model.
    conf_val : float
        The confidence value.
    num_bins : int
        The number of bins to use for the graph.

    Returns
    -------
    str
        The string representation of the node.

    """
    bin_val = math.ceil(conf_val * num_bins) / num_bins
    return f"{modelname}_{bin_val}"


def build_graph(
    output_dir: Path,
    num_bins: int = 10,
    min_connects: float = 1,
    cutoff: float = 0,
    upper_outlier_cutoff: float = 80,
    connectivity: int = 2,
    graph_type: type[nx.Graph | nx.DiGraph] = nx.Graph,
    *,
    purge_connectivity: bool | None = None,
) -> nx.Graph | nx.DiGraph:
    """
    Generate a graph from the model characterizations.

    Parameters
    ----------
    output_dir : Path
        The output directory to save the graph to.
        This directory should be the same as passed to the
        characterization.
    num_bins : int, optional
        The number of bins to use for the graph, by default 10.
    min_connects : float, optional
        The minimum number of connections an edge must have to not delete it
        By default 1.
    cutoff : float, optional
        The cutoff for the edge weights.
        By default 0.
    upper_outlier_cutoff : float, optional
        The cutoff for the upper outlier percentage.
        Any values in the top (100 - cutoff)% percent will be clamped.
        Helps to remove takeover edges.
        By default 80.
    connectivity : int, optional
        The minimum number of connected components the graph should have.
        Will only be checked if purge_connectivity is True.
        By default 2.
    graph_type : type[nx.Graph | nx.DiGraph], optional
        The type of networkx graph to use.
        By default nx.Graph.
        Options are: [nx.Graph, nx.DiGraph].
    purge_connectivity : bool | None, optional
        If True, the cutoff will be iteratively increased until the graph
        has the minimum number of connected components.
        By default None, which will not purge the graph.

    Returns
    -------
    nx.Graph | nx.DiGraph
        The graph of the model characterizations.

    Raises
    ------
    ValueError
        If an invalid metric is passed.

    """
    if purge_connectivity is None:
        purge_connectivity = False

    data_map: dict[str, dict[str, dict[str, str]]] = {}
    filenames_set: set[str] = set()
    modelnames_set: set[str] = set()
    num_models = 0
    num_entries = 0
    for root, dirs, _ in os.walk(output_dir):
        for modelname in dirs:
            modelnames_set.add(modelname)
            directory_path = Path(root) / modelname
            csv_path = str(directory_path.parts[-1]) + ".csv"
            with Path.open(directory_path / csv_path) as datafile:
                csv_data = csv.DictReader(datafile)
                data_map[modelname] = {}
                for row in csv_data:
                    filenames_set.add(row["filename"])
                    data_map[modelname][row["filename"]] = row
                    num_entries += 1
            num_models += 1
    # data_len = num_entries // num_models
    filenames: list[str] = sorted(filenames_set)
    modelnames: list[str] = sorted(modelnames_set)
    # connect the data together
    all_nodes: set[str] = set()
    all_edges: dict[str, set[str]] = defaultdict(set)
    all_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for file in filenames:
        conf_vals = {}
        for model in modelnames:
            data = data_map[model][file]
            conf_vals[model] = float(data["conf"])
        for model, conf_val in conf_vals.items():
            primary_node = get_node(model, conf_val, num_bins)
            all_nodes.add(primary_node)
            for model2, conf_val2 in conf_vals.items():
                if model == model2:
                    continue
                secondary_node = get_node(model2, conf_val2, num_bins)
                all_nodes.add(secondary_node)
                all_edges[primary_node].add(secondary_node)
                all_weights[primary_node][secondary_node] += 1

    # create the graph
    graph = create_graph(
        all_nodes=all_nodes,
        all_edges=all_edges,
        all_weights=all_weights,
        min_connects=min_connects,
        cutoff=cutoff,
        upper_outlier_cutoff=upper_outlier_cutoff,
        graph_type=graph_type,
    )
    graph_connectivity = len(list(nx.connected_components(graph)))

    # asses the connectivity of the graph
    if purge_connectivity:
        while graph_connectivity < connectivity:
            cutoff += 0.001
            graph = create_graph(
                all_nodes=all_nodes,
                all_edges=all_edges,
                all_weights=all_weights,
                min_connects=min_connects,
                cutoff=cutoff,
                upper_outlier_cutoff=upper_outlier_cutoff,
                graph_type=graph_type,
            )
            graph_connectivity = len(list(nx.connected_components(graph)))

    # write the graph to a file
    nx.write_weighted_edgelist(graph, output_dir / "conf_graph.txt")

    return graph


class ConfGraph:
    """Confidence graph wrapper class."""

    def __init__(
        self: Self,
        graph_file: Path | str | None = None,
    ) -> None:
        """
        Create a confidence graph wrapper instance.

        Parameters
        ----------
        graph_file : Path, str, optional
            The path to the created graph file.
            By default None, so it is required to build the graph
            using the build method.

        """
        self._graph: Path | None = (
            graph_file if graph_file is None else Path(graph_file)
        )
