"""
-------------------------------------------------------
vectorized utility and pseudo-surplus functions for network
formation models using TensorFlow
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import os
import tensorflow as tf
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


class VectorizedVStar(tf.Module):
    """
    -------------------------------------------------------
    A TensorFlow module that implements vectorized utility and pseudo-surplus
    calculations for network formation models. Supports both individual pairs
    and batch processing of node attributes and statistics.
    -------------------------------------------------------
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def U_star(self, xi, xj, si, sj, theta):
        """
        -------------------------------------------------------
        Calculates vectorized utility values for pairs of nodes based on their
        attributes and network statistics. Supports both individual pairs and batch processing.
        -------------------------------------------------------
        Parameters:
            xi - attributes of first nodes
                 For individual pairs: [n, x_dim] (tf.Tensor)
                 For batch processing: [unique_n, 1, x_dim] (tf.Tensor)
            xj - attributes of second nodes
                 For individual pairs: [n, x_dim] (tf.Tensor)
                 For batch processing: [n, 1, x_dim] (tf.Tensor)
            si - network statistics of first nodes
                 For individual pairs: [n] (tf.Tensor)
                 For batch processing: [unique_n, 1] (tf.Tensor)
            sj - network statistics of second nodes
                 For individual pairs: [n] (tf.Tensor)
                 For batch processing: [n, 1] (tf.Tensor)
            theta - model parameters [param_dim] (tf.Tensor)
        Returns:
            utility_values - matrix of utility values
                            For individual pairs: [n] (tf.Tensor)
                            For batch processing: [unique_n, n] (tf.Tensor)
        -------------------------------------------------------
        """
        unique_n = tf.shape(xi)[0]
        n = tf.shape(xj)[0]

        # Reshape theta to [param_dim, 1] for matmul
        theta_reshaped = tf.convert_to_tensor(theta, dtype=tf.float16)[:, tf.newaxis]

        # Concatenate inputs along the last dimension
        if len(xi.shape) == 3:  # Input should be [batch_size, x_dim]'
            # Reshape si, sj to match broadcasting dimensions
            si_expanded = tf.tile(tf.expand_dims(si, -1), [1, n, 1])  # [unique_n, n, 1]
            sj_expanded = tf.tile(tf.expand_dims(sj, 0), [unique_n, 1, 1])  # [unique_n, n, 1]
            xi_expanded = tf.tile(xi, [1, n, 1])  # [unique_n, n, x_dim]
            xj = tf.squeeze(xj, axis=1)  # Remove the middle dimension first: [n, x_dim]
            xj = tf.expand_dims(xj, 0)  # Add dimension at start: [1, n, x_dim]
            xj_expanded = tf.tile(xj, [unique_n, 1, 1])  # [unique_n, n, x_dim]

            # Concatenate features
            inputs = tf.concat([
                xi_expanded,
                xj_expanded,
                si_expanded,
                sj_expanded
            ], axis=-1)  # [unique_n, n, param_dim]
            assert len(inputs.shape) == 3 and inputs.shape[-1] == theta.shape[0], f'{inputs.shape=} {theta.shape[0]=}'
            inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])  # [unique_n*n, param_dim]

            # Perform dot product using matmul: [batch_size, param_dim] @ [param_dim, 1] -> [batch_size, 1]
            dot_products = tf.matmul(tf.cast(inputs, tf.float16), theta_reshaped)

            # Remove the last dimension and reshape to [unique_n, n]
            result = tf.reshape(dot_products, [unique_n, n])
        else:
            inputs = tf.concat([xi, xj, si[..., tf.newaxis], sj[..., tf.newaxis]], axis=-1)

            result = tf.matmul(tf.cast(inputs, tf.float16), theta_reshaped)

        return result

    @tf.function
    def V_star(self, xi, xj, si, sj, theta):
        """
        -------------------------------------------------------
        Calculates the pseudo-surplus function by summing utilities in both
        directions between node pairs. Supports both individual pairs and batch processing.
        -------------------------------------------------------
        Parameters:
            xi - attributes of first nodes
                 For individual pairs: [n, x_dim] (tf.Tensor)
                 For batch processing: [unique_n, 1, x_dim] (tf.Tensor)
            xj - attributes of second nodes
                 For individual pairs: [n, x_dim] (tf.Tensor)
                 For batch processing: [n, 1, x_dim] (tf.Tensor)
            si - network statistics of first nodes
                 For individual pairs: [n] (tf.Tensor)
                 For batch processing: [unique_n, 1] (tf.Tensor)
            sj - network statistics of second nodes
                 For individual pairs: [n] (tf.Tensor)
                 For batch processing: [n, 1] (tf.Tensor)
            theta - model parameters [param_dim] (tf.Tensor)
        Returns:
            surplus_values - matrix of pseudo-surplus values
                            For individual pairs: [n] (tf.Tensor)
                            For batch processing: [unique_n, n] (tf.Tensor)
        -------------------------------------------------------
        """
        # Compute both directions simultaneously
        forward = self.U_star(xi, xj, si, sj, theta)
        backward = tf.transpose(self.U_star(xj, xi, sj, si, theta))
        return forward + backward

    def __call__(self, xi, xj, si, sj, theta):
        """
        -------------------------------------------------------
        Main entry point for calculating pseudo-surplus values, handling both
        individual pairs and batches
        -------------------------------------------------------
        Parameters:
            xi - attributes of first nodes (array-like or tf.Tensor)
            xj - attributes of second nodes (array-like or tf.Tensor)
            si - network statistics of first nodes (array-like or tf.Tensor)
            sj - network statistics of second nodes (array-like or tf.Tensor)
            theta - model parameters (array-like or tf.Tensor)
        Returns:
            surplus_values - calculated pseudo-surplus values (tf.Tensor)
        -------------------------------------------------------
        """
        # If inputs are already tensors, use them directly
        if isinstance(xi, tf.Tensor):
            return self.V_star(xi, xj, si, sj, theta)

        # Convert individual inputs to tensors if needed
        xi = tf.convert_to_tensor(xi, dtype=tf.float16)
        xj = tf.convert_to_tensor(xj, dtype=tf.float16)
        si = tf.convert_to_tensor(si, dtype=tf.float16)
        sj = tf.convert_to_tensor(sj, dtype=tf.float16)
        theta = tf.convert_to_tensor(theta, dtype=tf.float16)

        return self.V_star(xi, xj, si, sj, theta)


def compute_all_v_values(unique_x: tf.Tensor, x_tensor: tf.Tensor, unique_s: tf.Tensor, s_tensor: tf.Tensor,
                         theta: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes pseudo-surplus values for all unique pairs of nodes in a batched manner
    using the VectorizedVStar module
    -------------------------------------------------------
    Parameters:
        unique_x - tensor of unique node attributes [unique_n, x_dim] (tf.Tensor)
        x_tensor - tensor of all node attributes [n, x_dim] (tf.Tensor)
        unique_s - tensor of network statistics for unique nodes [unique_n] (tf.Tensor)
        s_tensor - tensor of network statistics for all nodes [n] (tf.Tensor)
        theta - model parameters [param_dim] (tf.Tensor)
    Returns:
        v_values - matrix of pseudo-surplus values for all pairs [unique_n, n] (tf.Tensor)
                  where v_values[i,j] represents the surplus value between
                  unique node i and node j
    -------------------------------------------------------
    """

    x_unique = unique_x[:, tf.newaxis, :]  # [unique_n, 1, x_dim]
    s_unique = unique_s[:, tf.newaxis]  # [unique_n, 1]
    x_all = x_tensor[:, tf.newaxis, :]  # [n, 1, x_dim]
    s_all = s_tensor[:, tf.newaxis]  # [n, 1]

    return VectorizedVStar()(x_unique, x_all, s_unique, s_all, theta)
