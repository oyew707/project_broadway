"""
-------------------------------------------------------
implements a vectorized lookup function for H values in network formation models.
-------------------------------------------------------
Author:
    - Eric [xcz209@nyu.edu]
    - Alon  [abf386@nyu.edu]
    - Einstein Oyewole [eo2233@nyu.edu]
    - Anusha  [ad7038@nyu.edu]
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import os
import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


class VectorizedH:
    """
    -------------------------------------------------------
    A class implementing vectorized lookup functionality for H values in network
    formation models. Manages a mapping between node attributes and their
    corresponding H values, supporting batch operations via TensorFlow.
    -------------------------------------------------------
    Parameters:
        k_dim - number of dimensions for H values (int > 0)
        node_attrs - dictionary mapping node IDs to their attributes (dict)
        node_stats - dictionary mapping nodeIDS to node statistics(their degrees) (dict)
    -------------------------------------------------------
    """

    def __init__(self, k_dim, node_attrs, node_stats):
        assert k_dim > 0, "k_dim must be a positive integer"
        assert isinstance(node_attrs, dict), "node_attrs must be a dictionary"
        assert isinstance(node_stats, dict), "node_stats must be a dictionary"
        self.k_dim = k_dim
        self.node_attrs = node_attrs
        self.node_stats = node_stats
        self.lookup = self.default_h_values()
        log.debug(f'Default values: {repr(self)}')

    @staticmethod
    def generate_key(x):
        """
        -------------------------------------------------------
        Generates a string key from input tensor or array for the lookup table
        -------------------------------------------------------
        Parameters:
           x - input tensor or array to generate key from (tf.Tensor or array-like)
        Returns:
           key - string representation of input used as dictionary key (str)
        -------------------------------------------------------
        """
        if isinstance(x, tf.Tensor):
            return tf.strings.reduce_join(
                tf.as_string(x, precision=6),
                separator="_"
            ).numpy().decode()
        else:
            # Handle numpy arrays or lists
            return "_".join([f"{val:.6f}" for val in x])

    def default_h_values(self):
        """
        -------------------------------------------------------
        Initializes lookup dictionary with default H values based on average degree
        -------------------------------------------------------
        Returns:
           default_h - dictionary mapping node attribute keys to default H values (dict)
        -------------------------------------------------------
        """
        assert self.k_dim == 1, "k_dim must be 1 for default H values based on average degree"
        # Group Nodes b x values
        x_to_nodes = defaultdict(list)
        for node_id, x in self.node_attrs.items():
            x_key = self.generate_key(x)
            x_to_nodes[x_key].append(node_id)

        # Compute average degree for each x
        default_h = {}
        for x_key, nodes in x_to_nodes.items():
            degrees = [self.node_stats[node_id] for node_id in nodes]
            avg_degree = sum(degrees) / len(degrees)
            default_h[x_key] = tf.convert_to_tensor([avg_degree], dtype=tf.float16)

        return default_h

    def __repr__(self):
        """
        -------------------------------------------------------
        Creates detailed string representation of VectorizedH instance
        -------------------------------------------------------
        Returns:
           repr_str - detailed string representation including all H values (str)
        -------------------------------------------------------
        """
        lines = ["VectorizedH(k_dim={})".format(self.k_dim), "\nLookup Values:"]

        # Sort keys for consistent display
        sorted_keys = sorted(self.lookup.keys())
        for key in sorted_keys:
            value = self.lookup[key]
            # Convert tensor to numpy for cleaner printing
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            lines.append(f"  x={key}: H={value}")

        # Add some summary statistics
        lines.append("\nSummary:")
        lines.append(f"  Total unique x values: {len(self.lookup)}")
        lines.append(str(self))
        return "\n".join(lines)

    def __str__(self):
        """
        -------------------------------------------------------
        Creates concise string representation of VectorizedH instance
        -------------------------------------------------------
        Returns:
           str_repr - summary string with key instance information (str)
        -------------------------------------------------------
        """
        n_unique_x = len(self.lookup)

        # Calculate average H values
        avg_learned = np.mean([v.numpy() for v in self.lookup.values()]) if self.lookup else 0

        return (f"VectorizedH(k_dim={self.k_dim})\n"
                f"Unique x values: {n_unique_x}\n"
                f"Average H values: {avg_learned:.3f}")

    @staticmethod
    @tf.function
    def generate_keys_batch(x_batch):
        """
        -------------------------------------------------------
        Generates keys for a batch of input tensors
        -------------------------------------------------------
        Parameters:
           x_batch - batch of input tensors to generate keys for (tf.Tensor)
        Returns:
           keys - tensor of string keys for the input batch (tf.Tensor)
        -------------------------------------------------------
        """
        return tf.map_fn(
            lambda x: VectorizedH.generate_key(x),
            x_batch,
            dtype=tf.string
        )

    def update(self, mean_values):
        """
        -------------------------------------------------------
        Updates the lookup dictionary with new mean values
        -------------------------------------------------------
        Parameters:
           mean_values - dictionary containing new values to update lookup with (dict)

        -------------------------------------------------------
        """
        self.lookup = {k: tf.convert_to_tensor(v, dtype=tf.float16)
                       for k, v in mean_values.items()}

    @tf.function
    def __call__(self, x_batch):
        """
        -------------------------------------------------------
        Vectorized H function lookup for batches of x values
        -------------------------------------------------------
        Parameters:
           x_batch - batch of x values to lookup H values for ([batch_size, x_dim] tf.Tensor)
        Returns:
           H_values - corresponding H values for input batch ([batch_size, k_dim] tf.Tensor)
        -------------------------------------------------------
        """
        # Generate keys using static method
        x_strings = self.generate_keys_batch(x_batch)

        # Create a tensor of H values with k dimensions
        default_value = tf.zeros([self.k_dim], dtype=tf.float16)

        # Create a tensor of H values
        H_values = tf.map_fn(
            lambda x_str: tf.cast(
                tf.py_function(
                    lambda s: self.lookup.get(s.numpy().decode(), default_value),
                    inp=[x_str],
                    Tout=tf.float16
                ),
                dtype=tf.float16
            ),
            x_strings,
            fn_output_signature=tf.TensorSpec(shape=[self.k_dim], dtype=tf.float16)
        )

        return H_values

    def save(self, filepath: str):
        """
        -------------------------------------------------------
        Saves the VectorizedH instance's data to a pickle file
        -------------------------------------------------------
        Parameters:
            filepath - path to save the pickle file (str or Path)
        -------------------------------------------------------
        """
        save_dict = {
            'k_dim': self.k_dim,
            'lookup': self.lookup
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)
            log.info(f'Successfully saved VectorizedH to {filepath}')
        except Exception as e:
            log.warning(f'Failed to save VectorizedH to {filepath}: {str(e)}')
            pass

    def load(self, filepath: str):
        """
        -------------------------------------------------------
        Loads the lookup dictionary and k_dim from a pickle file and updates
        the VectorizedH instance
        -------------------------------------------------------
        Parameters:
            filepath - path to the pickle file to load (str or Path)
        -------------------------------------------------------
        """
        assert os.path.isfile(filepath), f'File {filepath} does not exist'

        try:
            with open(filepath, 'rb') as f:
                loaded_dict = pickle.load(f)

            # Verify the loaded dictionary has required keys
            required_keys = {'k_dim', 'lookup'}
            if not all(key in loaded_dict for key in required_keys):
                raise ValueError(f"Loaded file missing required keys: {required_keys}")

            # Update instance attributes
            self.k_dim = loaded_dict['k_dim']
            self.lookup = loaded_dict['lookup']

            log.info(f'Successfully loaded VectorizedH from {filepath}')

        except Exception as e:
            log.error(f'Failed to load VectorizedH from {filepath}: {str(e)}')
            raise 'Cannot load H'
