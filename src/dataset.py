"""
-------------------------------------------------------
Module to load and preprocess the dataset
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import os
import networkx as nx
import dsd
from collections import defaultdict
from src.likelihood import LikelihoodConfig
from typing import Tuple, Dict, List, Any, Set, Optional
import pandas as pd
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


def load_data(data_path: str = './data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    -------------------------------------------------------
    Load the NYSE dataset
    -------------------------------------------------------
    Parameters:
       data_path - path to the data files;
                    must contain nyse_node_sp1.csv, nyse_edge_buy_sp_sp1, nyse_edge_buy_com1.csv (str)
    Returns:
         node_data - node data (pd.DataFrame)
         edge_data - edge data (pd.DataFrame)
         committee_data - committee data (pd.DataFrame)
    -------------------------------------------------------
    """
    node_path = os.path.join(data_path, 'nyse_node_sp1.csv')
    edge_path = os.path.join(data_path, 'nyse_edge_buy_sp_sp1.csv')
    committee_path = os.path.join(data_path, 'nyse_edge_buy_com1.csv')

    assert os.path.exists(node_path), f"Node data file not found at {node_path}"
    assert os.path.exists(edge_path), f"Edge data file not found at {edge_path}"
    assert os.path.exists(committee_path), f"Committee data file not found at {committee_path}"

    # Read the data files
    node_data = pd.read_csv(node_path, header=None,
                            names=['name', 'ever_committee', 'node_id', 'ethnicity', 'ever_sponsor'])
    edge_data = pd.read_csv(edge_path, header=None,
                            names=['buyer_id', 'sponsor1_id', 'sponsor2_id', 'f1', 'f2', 'f3', 'f4', 'blackballs',
                                   'whiteballs', 'year'])
    committee_data = pd.read_csv(committee_path, header=None,
                                 names=['buyer_id', 'committee_id', 'f1', 'f2', 'f3', 'f4', 'blackballs', 'whiteballs',
                                        'year'])

    log.info("Data loaded successfully")
    log.debug(f"Node data: {node_data.shape}, Edge data: {edge_data.shape}, Committee data: {committee_data.shape}")

    return node_data, edge_data, committee_data


def process_data(node_data: pd.DataFrame, edge_data: pd.DataFrame, committee_data: pd.DataFrame) -> Tuple[
        Dict[int, List[float]], List[Dict[str, Any]], Dict[int, Dict[str, int]], Dict[int, Set[int]]]:
    """
    -------------------------------------------------------
    create node attributes, transaction records, network statistics, and edge connections for the NYSE sponsor network
    -------------------------------------------------------
    Parameters:
       node_data - DataFrame containing node information
                  (pd.DataFrame with columns: ['node_id', 'name', 'ethnicity',
                  'ever_committee', 'ever_sponsor'])
       edge_data - DataFrame containing edge information
                    (pd.DataFrame with columns: ['buyer_id', 'sponsor1_id',
                  'sponsor2_id', 'year', 'whiteballs', 'blackballs'])
       committee_data - DataFrame containing committee information for each transaction
                    (pd.DataFrame with columns: ['buyer_id', 'committee_id', 'year'])
    Returns:
        node_attrs - dictionary mapping node IDs to their attributes (dict)
        transactions - list of dictionaries containing transaction information (List[Dict])
        network_stats - dictionary mapping node IDs to degree and sponsor count (dict)
        edges - dictionary mapping node IDs to their connected nodes (dict)
    -------------------------------------------------------
    """
    # Check that all referenced IDs exist in node_data
    all_ids = set(node_data['node_id'])
    buyer_ids = set(edge_data['buyer_id'])
    sponsor_ids = set(edge_data['sponsor1_id']) | set(edge_data['sponsor2_id'])
    assert buyer_ids.issubset(all_ids), "Found buyer IDs not in node data"
    assert sponsor_ids.issubset(all_ids), "Found sponsor IDs not in node data"

    log.info("Processing data")
    # Convert ethnicity to one-hot encoding
    node_data = pd.get_dummies(data=node_data, columns=['ethnicity'], dummy_na=True, prefix='ethnicity',
                               drop_first=True, dtype=int)
    # If ever_committee and ever_sponsor are missing, fill with 0 (False)
    node_data[['ever_committee', 'ever_sponsor']] = node_data[['ever_committee', 'ever_sponsor']].fillna(0)
    # Set node_id as index
    node_attrs = node_data.set_index('node_id').drop(columns=['name'])
    log.info(f"Node attributes: {node_attrs.columns}")
    node_attrs = node_attrs.T.to_dict('list')

    # Initialize network statistics
    network_stats = {node_id: {'degree': 0, 'sponsor_count': 0} for node_id in node_attrs}
    edges = {node_id: set() for node_id in node_attrs}

    log.debug("Processing edge data")
    transactions = []
    # Process edge data
    for _, row in edge_data.iterrows():
        buyer_id = row['buyer_id']
        sponsor1_id = row['sponsor1_id']
        sponsor2_id = row['sponsor2_id']
        year = row['year']

        # Update network statistics
        network_stats[buyer_id]['degree'] += 2
        network_stats[sponsor1_id]['degree'] += 1
        network_stats[sponsor2_id]['degree'] += 1
        network_stats[sponsor1_id]['sponsor_count'] += 1
        network_stats[sponsor2_id]['sponsor_count'] += 1
        edges[buyer_id].add(sponsor1_id)
        edges[buyer_id].add(sponsor2_id)
        edges[sponsor1_id].add(buyer_id)
        edges[sponsor2_id].add(buyer_id)

        committee_members = committee_data[(committee_data['buyer_id'] == buyer_id) &
                                           (committee_data['year'] == year)]['committee_id'].tolist()

        transactions.append({
            'buyer_id': buyer_id,  # Node ID of the buyer
            'sponsor1_id': sponsor1_id,  # Node ID of the first sponsor
            'sponsor2_id': sponsor2_id,  # Node ID of the second sponsor
            'committee_members': committee_members,
            'year': year,
            'whiteballs': row['whiteballs'],  # Number of positive votes by committee members
            'blackballs': row['blackballs']  # Number of negative votes by committee members
        })
    log.info("Data processed successfully")

    return node_attrs, transactions, network_stats, edges


def most_connected_subgraph(likelihood_config: LikelihoodConfig) -> nx.Graph:
    """
    -------------------------------------------------------
    Finds the most connected subgraph in the network.
    -------------------------------------------------------
    Parameters:
       likelihood_config - Original configuration with full network data (LikelihoodConfig)
    Returns:
         subgraph - Most connected subgraph (nx.Graph)
    -------------------------------------------------------
    """
    # Convert the graph to networkx
    G = nx.Graph()
    for node, neighbors in likelihood_config.edges.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    log.info(f'Extracting most connected subgraph')
    log.debug(f'Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    # Find the most connected subgraph
    subgraph = max(nx.connected_components(G), key=len)
    log.info(f'Extracted subgraph with {len(subgraph)} nodes')

    return G.subgraph(subgraph)


def densest_subgraph(likelihood_config: Optional[LikelihoodConfig], graph: Optional[nx.Graph]) -> nx.Graph:
    """
    -------------------------------------------------------
    Finds the densest subgraph in the network.
    -------------------------------------------------------
    Parameters:
       likelihood_config - Original configuration with full network data (LikelihoodConfig)
       graph - Networkx graph to find the densest subgraph in (nx.Graph)
    Returns:
         subgraph - densest subgraph (nx.Graph)
    -------------------------------------------------------
    """
    assert (likelihood_config is not None) or (graph is not None), "Either likelihood_config or G must be provided"

    if graph is None:
        # Convert the graph to networkx
        G = nx.Graph()
        for node, neighbors in likelihood_config.edges.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    else:
        # Convert the graph to networkx
        G = graph.copy()

    log.info(f'Extracting densest subgraph')
    # Find the densest subgraph
    subgraph, density = dsd.exact_densest(G)
    log.info(f'Extracted subgraph with Density: {density} and with {len(subgraph)} nodes')

    return G.subgraph(subgraph)


def graph_to_config(graph: nx.Graph, likelihood_config) -> LikelihoodConfig:
    """
    -------------------------------------------------------
    Converts a graph to a configuration object.
    -------------------------------------------------------
    Parameters:
         graph - Graph to convert (nx.Graph)
    Returns:
       subgraph_config - Configuration object (LikelihoodConfig)
    -------------------------------------------------------
    """
    # Create new config with only nodes in largest component
    subgraph_edges = {n: set(graph.neighbors(n)) for n in graph.nodes()}

    subgraph_config = LikelihoodConfig(
        node_attrs={n: likelihood_config.node_attrs[n] for n in graph.nodes()},
        node_stats={n: likelihood_config.node_stats[n] for n in graph.nodes()},
        weights={n: likelihood_config.weights[n] for n in graph.nodes()},
        edges=subgraph_edges
    )

    return subgraph_config