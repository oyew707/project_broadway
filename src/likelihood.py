"""
-------------------------------------------------------
Implements likelihood computations for network formation models,
including H* computation, importance weights calculation, and log-likelihood
evaluations.
-------------------------------------------------------
Author:
    - Eric [xcz209@nyu.edu]
    - Alon  [abf386@nyu.edu]
    - Einstein Oyewole [eo2233@nyu.edu]
    - Anusha  [ad7038@nyu.edu]
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Import
import copy
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict,  Optional, Iterable
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from src.H import VectorizedH
from src.Psi import compute_psi_values
from src.VStar import VectorizedVStar, compute_all_v_values
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


@dataclass
class LikelihoodConfig:
    """
    -------------------------------------------------------
    Configuration for likelihood computation
    -------------------------------------------------------
    Parameters:
       node_attrs - dictionary mapping node IDs to attribute vectors (dict)
       node_stats - dictionary mapping node IDs to network statistics (dict)
       weights - dictionary mapping node IDs to node weights (dict)
       edges - dictionary mapping node IDs to sets of connected nodes (dict)
    -------------------------------------------------------
    """
    node_attrs: Dict
    node_stats: Dict
    weights: Dict
    edges: Dict


def H_star(node_attrs: Dict, node_stats: Dict, weights: Dict, theta: tf.Tensor, H: VectorizedH,
           max_iter: int = 20) -> VectorizedH:
    """
    -------------------------------------------------------
    Computes the H* function through an iterative fixed-point computation for
    network formation models.
    -------------------------------------------------------
    Parameters:
       node_attrs - dictionary mapping node IDs to attribute vectors (dict)
       node_stats - dictionary mapping node IDs to network statistics (dict)
       weights - dictionary mapping node IDs to node weights (dict)
       theta - model parameters (tf.Tensor)
       H - vectorized H function instance (VectorizedH)
       max_iter - maximum number of iterations for convergence (int > 0)
    Returns:
       H_converged - converged H function after fixed point iteration (VectorizedH)
    -------------------------------------------------------
    """
    assert max_iter > 0, "max_iter must be a positive integer"

    log.info("Computing H*")
    # initialize tensors for vectorized processing
    H_current = H
    x_tensor = tf.convert_to_tensor(np.array([node_attrs[i] for i in node_attrs.keys()]), dtype=tf.float16)
    s_tensor = tf.convert_to_tensor(np.array([node_stats[i] for i in node_attrs.keys()]), dtype=tf.float16)
    w_tensor = tf.convert_to_tensor(np.array([weights[i] for i in node_attrs.keys()]), dtype=tf.float16)
    unique_node_pairs = set((tuple(node_attrs[node_id]), node_stats[node_id])
                            for node_id in node_attrs.keys())
    unique_x = tf.convert_to_tensor([x for x, s in unique_node_pairs], dtype=tf.float16)
    unique_s = tf.convert_to_tensor([s for x, s in unique_node_pairs], dtype=tf.float16)

    x_group = defaultdict(list)
    # Grouped indices for each unique x
    for idx, (x, _) in enumerate(unique_node_pairs):
        x_key = VectorizedH.generate_key(x)
        x_group[x_key].append(idx)

    V = compute_all_v_values(unique_x, x_tensor, unique_s, s_tensor, theta)
    log.debug(f'Computed {V=} \n Using {theta=}')

    for i in tqdm(range(max_iter), desc='H_star', position=0, leave=True):
        H_prev = copy.deepcopy(H_current)

        psi_values = compute_psi_values(unique_x, x_tensor, w_tensor, s_tensor, H_prev, V)
        # Compute means for each unique x
        psi_mean = {}
        for x, indices in x_group.items():
            selected_psi_values = tf.gather(psi_values, indices, axis=0)
            psi = tf.reduce_mean(tf.cast(selected_psi_values, tf.float32), axis=[0, 1])
            psi_mean[x] = tf.cast(psi, tf.float16)

        # Update H lookup
        H_current.update(psi_mean)

        # Check convergence - now comparing full tensors
        diff = tf.stop_gradient(tf.reduce_max(tf.abs(H_current(unique_x) - H_prev(unique_x))))
        if diff < 1e-4:
            log.info(f"Convergence achieved after {i} iterations.")
            break

    return H_current


@lru_cache(maxsize=1000)
def calculate_weights(s: float, node_vals: Iterable) -> float:
    """
    -------------------------------------------------------
    Calculates importance weights for nodes based on network statistics.
    Implements importance weight function from section 4.2.1 pg 36.
    -------------------------------------------------------
    Parameters:
       s - network statistic value (float)
       node_vals - collection of node values (iterable)
    Returns:
       weights - calculated importance weight (float)
    -------------------------------------------------------
    """
    t = tf.convert_to_tensor(list(node_vals))
    indicator_tensor = tf.cast(tf.greater(t, 0), tf.float16)
    denominator = tf.reduce_mean(indicator_tensor).numpy()
    return s / denominator


def vectorized_log_likelihood_contribution(Lijt: tf.Tensor, x_i: tf.Tensor, x_j: tf.Tensor, s_i: tf.Tensor,
                                           s_j: tf.Tensor, theta: tf.Tensor, H: VectorizedH) -> tf.Tensor:
    """
    -------------------------------------------------------
    Calculates vectorized log-likelihood contributions for individual pairs of nodes.
    -------------------------------------------------------
    Parameters:
       Lijt - tensor of link indicators [batch_size] (tf.Tensor)
       x_i - attributes of first nodes [batch_size, x_dim] (tf.Tensor)
       x_j - attributes of second nodes [batch_size, x_dim] (tf.Tensor)
       s_i - network statistics of first nodes [batch_size] (tf.Tensor)
       s_j - network statistics of second nodes [batch_size] (tf.Tensor)
       theta - model parameters (tf.Tensor)
       H - function mapping attributes to H values (VectorizedH)
    Returns:
       ll_contributions - log-likelihood contributions [batch_size] (tf.Tensor)
    -------------------------------------------------------
    """
    # Compute V* for all pairs at once
    V = VectorizedVStar()(x_i, x_j, s_i, s_j, theta)

    # Compute H values for all nodes at once
    H_i = tf.squeeze(H(x_i))
    H_j = tf.squeeze(H(x_j))

    # Compute log likelihood contributions vectorized
    ll = 0.5 * Lijt * (V - tf.math.log1p(H_i) - tf.math.log1p(H_j))
    return ll


def log_likelihood_optimized(theta: tf.Tensor, likelihood_config: LikelihoodConfig,
                             H: Optional[VectorizedH] = None, use_mean: bool = False) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes the log-likelihood for the network formation model using
    vectorized operations.
    Note: Normalizes the log likelihood contributions by the number of edges and nodes for numerical stability.
    -------------------------------------------------------
    Parameters:
       theta - model parameters (tf.Tensor)
       likelihood_config - configuration for likelihood computation with node_attrs, node_stats, weights, edges (LikelihoodConfig)
       H - optional pre-computed H function (VectorizedH or None)
       use_mean - whether to use mean or sum for normalization (bool)
    Returns:
       ll - computed log-likelihood value (tf.Tensor)
       H - updated H function after computation (VectorizedH)
    -------------------------------------------------------
    """
    # cast theta to less precision (memory issues and whatnot)
    theta = tf.cast(theta, tf.float16)

    # # initialize H function and attribute tensors
    H = H if H is not None else VectorizedH(k_dim=1, node_attrs=likelihood_config.node_attrs,
                                            node_stats=likelihood_config.node_stats)
    H = H_star(likelihood_config.node_attrs, likelihood_config.node_stats, likelihood_config.weights, theta, H)
    x_i, x_j, s_i, s_j = [], [], [], []
    for i in likelihood_config.node_attrs.keys():
        for j in likelihood_config.edges[i]:
            if i == j:
                continue
            x_i.append(likelihood_config.node_attrs[i])
            x_j.append(likelihood_config.node_attrs[j])
            s_i.append(likelihood_config.node_stats[i])
            s_j.append(likelihood_config.node_stats[j])
    x_i = tf.convert_to_tensor(x_i, dtype=tf.float16)
    x_j = tf.convert_to_tensor(x_j, dtype=tf.float16)
    s_i = tf.convert_to_tensor(s_i, dtype=tf.float16)
    s_j = tf.convert_to_tensor(s_j, dtype=tf.float16)
    Lijt = tf.ones(s_i.shape, dtype=tf.float16)

    # compute likelihood for all edges at once
    edge_ll = vectorized_log_likelihood_contribution(
        Lijt, x_i, x_j, s_i, s_j, theta, H,
    )

    # Sum edge contributions
    log.debug(f"Is NA or infinity: {tf.reduce_any(tf.math.is_nan(edge_ll)) or tf.reduce_any(tf.math.is_inf(edge_ll))}")
    # Normalize by number of edges
    ll = tf.reduce_mean(tf.cast(edge_ll, tf.float32)) if use_mean else tf.reduce_sum(tf.cast(edge_ll, tf.float32))
    log.debug(f'Log likelihood: {ll}')

    # Add node-specific terms
    node_term = tf.math.log(tf.squeeze(s_i)) - tf.math.log1p(tf.squeeze(H(x_i)))
    # Note node_terms should exist because we only consider s_i > 0 and 1 + H > 0
    # Normalize by number of nodes
    ll += tf.reduce_mean(tf.cast(node_term, tf.float32)) if use_mean else tf.reduce_sum(tf.cast(node_term, tf.float32))
    # return tf.math.reduce_sum(theta), H

    return ll, H
