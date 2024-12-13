"""
-------------------------------------------------------
Implements the potential value function for network formation models.
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "12/13/24"
-------------------------------------------------------
"""

# Imports
import os
from collections import defaultdict

from src.logger import getlogger
from src.H import VectorizedH
import tensorflow as tf
import numpy as np

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


@tf.function
def compute_M_star(x_tensor: tf.Tensor, s_tensor: tf.Tensor, H: VectorizedH) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes the potential value distribution M*₁ for network statistics
    given node attributes and H values.
    Assumes there is no endogenous interaction effect
    H*(x₁)^s₁₁ / (1 + H*(x₁))^(s₁₁+1)  ... pg22
    -------------------------------------------------------
    Parameters:
       x_tensor - node attributes [n, x_dim] (tf.Tensor)
        s_tensor - network statistics [n] (tf.Tensor)
        H - vectorized H function (VectorizedH)
    Returns:
       M_star - potential value distribution for network statistics [n] (tf.Tensor)
    -------------------------------------------------------
    """
    # Get H values for attributes
    H_values = H(x_tensor)  # [batch_size, k_dim]

    # Compute components
    numerator = tf.pow(H_values, s_tensor)  # H*(x)^s 
    denominator = tf.pow(1 + H_values, s_tensor + 1)  # (1 + H*(x))^(s+1)

    # Compute distribution
    M = numerator / denominator

    return M


@tf.function
def compute_general_M_star(x_tensor: tf.Tensor, s_tensor: tf.Tensor, H: VectorizedH,
                            theta: tf.Tensor, max_iter: int = 20) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes the potential value distribution M* for the general case
    with strategic interaction effects.
    TODO: Test and confirm implementation
    -------------------------------------------------------
    Parameters:
       x_tensor - node attributes [n, x_dim] (tf.Tensor)
        s_tensor - network statistics [n] (tf.Tensor)
        H - vectorized H function (VectorizedH)
        theta - model parameters (tf.Tensor)
        max_iter - maximum number of iterations for convergence (int > 0)
    Returns:
       M_star - potential value distribution for network statistics [n] (tf.Tensor)
    -------------------------------------------------------
    """
    # Initialize with simple case
    M = compute_M_star(x_tensor, s_tensor, H)

    # Fixed point iteration
    for _ in range(max_iter):
        M_prev = tf.identity(M)

        # Compute Omega₀ mapping based on strategic effects
        # This depends on specific model structure
        M = compute_omega_mapping(x_tensor, s_tensor, H, theta, M_prev)

        # Check convergence
        if tf.reduce_max(tf.abs(M - M_prev)) < 1e-6:
            break


@tf.function
def compute_degree_probability(x: tf.Tensor, r: tf.Tensor, H: VectorizedH) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes probability of a node having degree r given attributes x
    -------------------------------------------------------
    Parameters:
        x - node attributes [batch_size, x_dim] (tf.Tensor)
        r - degree values to compute prob for [batch_size] (tf.Tensor)
        H - vectorized H function with computed H* values (VectorizedH)
    Returns:
        probs - probability of degree r for each node [batch_size] (tf.Tensor)
    -------------------------------------------------------
    """
    # Get H*(x) values
    H_values = H(x)  # [batch_size, 1]

    # Cast degree to float for exponentiation
    r = tf.cast(r, tf.float32)

    # Compute numerator: H*(x)^r
    numerator = tf.pow(H_values, r)

    # Compute denominator: (1 + H*(x))^(r+1)
    denominator = tf.pow(1 + H_values, r + 1)

    # Compute probability
    probs = numerator / denominator

    return probs


@tf.function
def update_potential_values(x_tensor: tf.Tensor, s_tensor: tf.Tensor, H: VectorizedH) -> Dict:
    """
    -------------------------------------------------------
    Updates the potential value distribution M* using degree probabilities
    for the case with no endogenous interaction effects
    -------------------------------------------------------
    Parameters:
        x_tensor - node attributes [n, x_dim] (tf.Tensor)
        s_tensor - observed network statistics/degrees [n] (tf.Tensor)
        H - vectorized H function (VectorizedH)
    Returns:
        potential_values - updated potential value distribution [n] (tf.Tensor)
    -------------------------------------------------------
    """
    # Compute degree probabilities for each unique x value
    x_to_nodes = defaultdict(list)

    # Group nodes by x values
    for i in range(tf.shape(x_tensor)[0]):
        x_key = VectorizedH.generate_key(x_tensor[i])
        x_to_nodes[x_key].append(i)

    # Compute M* for each unique x
    potential_values = {}
    for x_key, node_indices in x_to_nodes.items():
        # Get attributes and degrees for this group
        x = x_tensor[node_indices[0]]  # Take first instance of x
        s = tf.gather(s_tensor, node_indices)

        # For no endogenous effects, M* equals degree probability
        M_x = compute_degree_probability(
            tf.expand_dims(x, 0),  # Add batch dimension
            s,
            H
        )

        potential_values[x_key] = M_x

    return tf.convert_to_tensor(potential_values.values())


@tf.function
def compute_omega_mapping(x: tf.Tensor, s: tf.Tensor, H: VectorizedH,
                          theta: tf.Tensor, M: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes the Omega mapping for potential value distribution M
    -------------------------------------------------------
    Parameters:
         x - node attributes [n, x_dim] (tf.Tensor)
         s - network statistics [n] (tf.Tensor)
         H - vectorized H function (VectorizedH)
         theta - model parameters (tf.Tensor)
         M - current potential value distribution [n] (tf.Tensor)
    Returns:
            M_new - updated potential value distribution [n] (tf.Tensor)
    -------------------------------------------------------
    """

    # Example implementation for degree complementarities
    H_values = H(x)

    # Compute degree distribution under current M
    degree_probs = compute_degree_probability(x, s, H)

    # Update M based on strategic effects
    # Specific form depends on model
    M_new = update_potential_values(degree_probs, theta)

    return M_new
