"""
-------------------------------------------------------
implements vectorized PSI (Pairwise Strategic Interaction)
calculations for network formation models using TensorFlow.
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from src.H import VectorizedH
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, 'debug')
EPSILON = 1e-6
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
MAX_EXP = 10


class VectorizedPsi(tf.Module):
    """
    -------------------------------------------------------
    Implements vectorized PSI calculations for network formation models.
    -------------------------------------------------------
    """

    def __init__(self):
        super().__init__()

    def __call__(self, s_batch: tf.Tensor, w_batch: tf.Tensor, V_batch: tf.Tensor,
                 H_batch: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes vectorized PSI values for batches of network data.
        ψ = w * s * exp(V) / (1 + H)
        -------------------------------------------------------
        Parameters:
            s_batch - batch of network statistics [unique_n, n] (tf.Tensor)
            w_batch - batch of node weights/probabilities [unique_n, n] (tf.Tensor)
            V_batch - batch of utility values [unique_n, n] (tf.Tensor)
            H_batch - batch of H values [unique_n, n, k_dim] (tf.Tensor)
        Returns:
            psi_values - computed PSI values for all node pairs and dimensions
                        [unique_n, n, k_dim] (tf.Tensor)
        -------------------------------------------------------
        """
        unique_n, n = tf.shape(s_batch)[0], tf.shape(s_batch)[1]
        k_dim = tf.shape(H_batch)[2]

        # Flatten the inputs
        s_flat = tf.reshape(s_batch, [-1])  # [unique_n*n]
        w_flat = tf.reshape(w_batch, [-1])  # [unique_n*n]
        V_flat = tf.reshape(V_batch, [-1])  # [unique_n*n]
        H_values = tf.reshape(H_batch, [-1, k_dim])  # [unique_n*n, k]

        # Compute numerator
        num = tf.exp(tf.minimum(V_flat, MAX_EXP))[..., tf.newaxis]  # [unique_n*n, 1]
        if tf.reduce_any(tf.math.is_nan(num)) or tf.reduce_any(tf.math.is_inf(num)):
            is_nan_or_inf = tf.math.is_nan(num) | tf.math.is_inf(num)
            indices = tf.where(is_nan_or_inf)[..., 0]
            log.error(f' {tf.gather(V_flat, indices)=}')
            log.error(f'{tf.gather(num, indices)=}')
            raise ValueError('NAN when computing psi: numerator')

        # Compute denominator
        denom = 1 + H_values  # [unique_n*n, k]

        # Expand dimensions for broadcasting
        s_flat = s_flat[..., tf.newaxis]  # [unique_n*n, 1]
        w_flat = w_flat[..., tf.newaxis]  # [unique_n*n, 1]

        # Compute final result
        res_flat = tf.cast(w_flat, tf.float32) * tf.cast(s_flat, tf.float32) * (tf.cast(num, tf.float32) / tf.cast(denom, tf.float32))
        # res_flat = tf.cast(res_flat, tf.float16)  # [unique_n*n, k]
        if tf.reduce_any(tf.math.is_nan(res_flat)) or tf.reduce_any(tf.math.is_inf(res_flat)):
            is_nan_or_inf = tf.math.is_nan(res_flat) | tf.math.is_inf(res_flat)
            indices = tf.where(is_nan_or_inf)[..., 0]
            log.error(f'{tf.gather(w_flat, indices)=}')
            log.error(f'{tf.gather(s_flat, indices)=}')
            log.error(f'{tf.gather(num, indices)=}')
            log.error(f'{tf.gather(H_values, indices)=}')
            raise ValueError('NAN when computing psi: all results')

        # Reshape back to original dimensions
        res = tf.reshape(res_flat, [unique_n, n, -1])  # [unique_n, n, k]
        return res


def compute_psi_values(unique_x: tf.Tensor, x_tensor: tf.Tensor, w_tensor: tf.Tensor, s_tensor: tf.Tensor,
                       H: VectorizedH, V_values: tf.Tensor):
    """
    -------------------------------------------------------
    Computes Pairwise Strategic Interaction (PSI) values for all unique node pairs
    using vectorized operations. Reshapes and Broadcasts inputs to enable batch
    computation of strategic effects.
    -------------------------------------------------------
    Parameters:
        unique_x - tensor of unique node attributes [unique_n, x_dim] (tf.Tensor)
        x_tensor - tensor of all node attributes [n, x_dim] (tf.Tensor)
        w_tensor - tensor of node weights/probabilities [n] (tf.Tensor)
        s_tensor - tensor of network statistics [n] (tf.Tensor)
        H - function object for computing H values (VectorizedH)
        V_values - pre-computed pseudo-surplus values [unique_n * n] (tf.Tensor)
    Returns:
        psi_values - tensor of PSI values for all pairs [unique_n, n, k_dim=1] (tf.Tensor)
                    where:
                    - first dimension indexes unique nodes
                    - second dimension indexes all nodes
                    - third dimension contains k_dim PSI values per pair
    -------------------------------------------------------
    """
    n_unique = tf.shape(unique_x)[0]
    n_nodes = tf.shape(x_tensor)[0]

    H_values = H(unique_x)

    # Reshape inputs for broadcasting
    x_all = tf.tile(x_tensor[tf.newaxis, ...], [n_unique, 1, 1])  # [unique_n, n, x_dim]
    s_all = tf.tile(s_tensor[tf.newaxis, ...], [n_unique, 1])  # [unique_n, n]
    w_all = tf.tile(w_tensor[tf.newaxis, ...], [n_unique, 1])  # [unique_n, n]
    h_all = tf.tile(H_values[tf.newaxis, ...], [1, n_nodes, H.k_dim])  # [unique_n, n, k]

    # Reshape V values to match
    V_reshaped = tf.reshape(V_values, [n_unique, n_nodes])  # [unique_n, n]

    # Compute PSI values for all pairs
    res = VectorizedPsi()(s_all, w_all, V_reshaped, h_all)  # [unique_n, n]
    return res
