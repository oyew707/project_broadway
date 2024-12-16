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
from collections import defaultdict
from typing import Tuple
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
        dict, list, dict, dict]:
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
    edges = defaultdict(set)

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
