"""
-------------------------------------------------------
Generates likely graph/networks configuration based on
the data and model
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "12/12/24"
-------------------------------------------------------
"""


# Imports
import os
from src.H import VectorizedH
from src.M import compute_M_star
from src.VStar import VectorizedVStar
from src.likelihood import LikelihoodConfig
from src.logger import getlogger
import tensorflow as tf
import numpy as np

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)
SEED = int(os.getenv('RANDOMSEED', '4'))
tf.random.set_seed(SEED)
MAX_EXP = 10


def sample_graph(likelihood_config: LikelihoodConfig, theta: tf.Tensor, H: VectorizedH) -> tf.Tensor:
    """
    -------------------------------------------------------
    Samples a likely network configuration based on the data and an
    optimized model parameter
    -------------------------------------------------------
    Parameters:
        likelihood_config - configuration for likelihood computation (LikelihoodConfig)
        theta - optimized/sampled model parameters (tf.Tensor)
        H - vectorized H function (VectorizedH)
    Returns:
        likely_graph - adjacency matrix of the likely network (tf.Tensor [num_nodes, num_nodes])
    -------------------------------------------------------
    """
    assert isinstance(likelihood_config, LikelihoodConfig), "likelihood_config must be LikelihoodConfig instance"
    assert isinstance(theta, tf.Tensor), "theta must be a Tensor"
    assert isinstance(H, VectorizedH), "H must be a VectorizedH instance"

    # Create tensors for node attributes and statistics
    x_i = tf.convert_to_tensor(np.array([likelihood_config.node_attrs[i] for i in likelihood_config.node_attrs.keys()]), dtype=tf.float16)
    x_j = tf.identity(x_i)
    s_i = tf.convert_to_tensor(np.array([likelihood_config.node_stats[i] for i in likelihood_config.node_attrs.keys()]), dtype=tf.float16)
    s_j = tf.identity(s_i)

    # Broadcast for batch processing
    x_i = tf.expand_dims(x_i, axis=1)
    x_j = tf.expand_dims(x_j, axis=1)
    s_i = tf.expand_dims(s_i, axis=1)
    s_j = tf.expand_dims(s_j, axis=1)

    # Compute V* values and H functions
    V = VectorizedVStar()(x_i, x_j, s_i, s_j, theta)
    H_i = tf.squeeze(H(x_i))
    H_j = tf.squeeze(H(x_j))

    # Broadcast H values
    H_i = tf.tile(tf.expand_dims(H_i, axis=1), [1, x_j.shape[1]])
    H_j = tf.tile(tf.expand_dims(H_j, axis=1), [1, x_i.shape[1]])

    log.debug(f'{V.shape=} {H_i.shape=} {H_j.shape=}')
    # Sample edge with probability proportional to exp(V)/(1+H_i)(1+H_j)
    denominator = tf.cast((1 + H_i) * (1 + H_j), tf.float32)
    numerator = tf.cast(tf.exp(tf.minimum(V, MAX_EXP)), tf.float32)
    p_ij = numerator / denominator
    log.debug(f'{p_ij.shape=}, Min: {tf.reduce_min(p_ij)}, Max: {tf.reduce_max(p_ij)}')
    # clip probabilities to avoid numerical issues
    p_ij = tf.cast(tf.clip_by_value(p_ij, 1e-7, 1.0), tf.float16)

    # L = tf.random.stateless_bernoulli(SEED, probs=p_ij, dtype=tf.int32)
    L = tf.random.stateless_uniform(p_ij.shape, seed=[SEED, SEED], minval=0, maxval=1, dtype=tf.float16) < p_ij
    L = tf.cast(L, tf.int32)
    # Make symmetric by taking upper triangle
    L_upper = tf.linalg.band_part(L, 0, -1)  # Upper triangle including diagonal
    L_no_diag = L_upper - tf.linalg.diag(tf.linalg.diag_part(L_upper))  # Remove diagonal
    L_symmetric = L_no_diag + tf.transpose(L_no_diag)  # Make symmetric

    return L_symmetric

@tf.function
def limiting_link_intensity(x_i: tf.Tensor, x_j: tf.Tensor, s_i: tf.Tensor, s_j: tf.Tensor, H: VectorizedH, theta: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Limiting link intensity for a given network configuration
    f = s1 * s2 * exp(V) / (1 + H1)(1 + H2)
    See pg 21
    -------------------------------------------------------
    Parameters:
        x_i - node attributes [n, x_dim] (tf.Tensor)
        x_j - node attributes [n, x_dim] (tf.Tensor)
        s_i - observed network statistics/degrees [n] (tf.Tensor)
        s_j - observed network statistics/degrees [n] (tf.Tensor)
        H - vectorized H function (VectorizedH)
        theta - model parameters (tf.Tensor)
    Returns:
        f - link intensity [n, n] (tf.Tensor)
    -------------------------------------------------------
    """
    # Compute H values for all nodes at once
    H_i = tf.squeeze(H(x_i))
    H_j = tf.squeeze(H(x_j))

    # Broadcast for batch processing
    x_i = tf.expand_dims(x_i, axis=1)
    x_j = tf.expand_dims(x_j, axis=1)
    s_i = tf.expand_dims(s_i, axis=1)
    s_j = tf.expand_dims(s_j, axis=1)

    # compute m*
    m_i = compute_M_star(x_i, s_i, H)
    m_j = compute_M_star(x_j, s_j, H)

    # Compute V* for all pairs at once
    V = VectorizedVStar()(x_i, x_j, s_i, s_j, theta)

    H_i = tf.tile(tf.expand_dims(H_i, axis=1), [1, x_j.shape[1]])
    H_j = tf.tile(tf.expand_dims(H_j, axis=1), [1, x_i.shape[1]])
    s_i = tf.cast(tf.tile(s_i, [1, x_j.shape[1]]), tf.float32)
    s_j = tf.cast(tf.tile(s_j, [1, x_i.shape[1]]), tf.float32)
    m_i = tf.cast(tf.tile(m_i, [1, x_j.shape[1]]), tf.float32)
    m_j = tf.cast(tf.tile(m_j, [1, x_i.shape[1]]), tf.float32)

    # Cast to float32 for numerical stability
    denominator = tf.cast((1 + H_i) * (1 + H_j), tf.float32)
    numerator = tf.cast(tf.exp(tf.minimum(V, MAX_EXP)), tf.float32)

    # Compute link intensity
    f = s_i * s_j * m_i * m_j * numerator / denominator
    return tf.cast(f, tf.float16)


@tf.function
def sample_limiting_distribution(likelihood_config: LikelihoodConfig, theta: tf.Tensor, H: VectorizedH) -> tf.Tensor:
    """
    -------------------------------------------------------
    Sample a network configuration from the limiting distribution
    -------------------------------------------------------
    Parameters:
       likelihood_config - configuration for likelihood computation (LikelihoodConfig)
        theta - optimized/sampled model parameters (tf.Tensor [num_dims_theta])
        H - vectorized H function (VectorizedH)
    Returns:
       likely_graph - adjacency matrix of the likely network (tf.Tensor [num_nodes, num_nodes])
    -------------------------------------------------------
    """
    assert isinstance(likelihood_config, LikelihoodConfig), "likelihood_config must be LikelihoodConfig instance"
    assert isinstance(theta, tf.Tensor), "theta must be a Tensor"
    assert isinstance(H, VectorizedH), "H must be a VectorizedH instance"
    # Create tensors for node attributes and statistics
    x_i = tf.convert_to_tensor(np.array([likelihood_config.node_attrs[i] for i in likelihood_config.node_attrs.keys()]), dtype=tf.float16)
    x_j = tf.identity(x_i)
    s_i = tf.convert_to_tensor(np.array([likelihood_config.node_stats[i] for i in likelihood_config.node_attrs.keys()]), dtype=tf.float16)
    s_j = tf.identity(s_i)

    p_ij = limiting_link_intensity(x_i, x_j, s_i, s_j, H, theta)
    log.debug(f'{p_ij.shape=}, Min: {tf.reduce_min(p_ij)}, Max: {tf.reduce_max(p_ij)}')
    # clip probabilities to avoid numerical issues
    p_ij = tf.clip_by_value(p_ij, 1e-7, 1.0)

    # L = tf.random.stateless_bernoulli(SEED, probs=p_ij, dtype=tf.int32)
    L = tf.random.stateless_uniform(p_ij.shape, seed=[SEED, SEED], minval=0, maxval=1, dtype=tf.float16) < p_ij
    L = tf.cast(L, tf.int32)
    # Make symmetric by taking upper triangle
    L_upper = tf.linalg.band_part(L, 0, -1)  # Upper triangle including diagonal
    L_no_diag = L_upper - tf.linalg.diag(tf.linalg.diag_part(L_upper))  # Remove diagonal
    L_symmetric = L_no_diag + tf.transpose(L_no_diag)  # Make symmetric

    return L_symmetric




