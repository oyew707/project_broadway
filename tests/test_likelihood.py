"""
-------------------------------------------------------
Test suite for likelihood calculations in network formation models
-------------------------------------------------------
Author: einsteinoyewole
Email:  eo2233@nyu.edu
__updated__ = "12/12/24"
-------------------------------------------------------
"""

# Imports
import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch
from src.Psi import compute_psi_values
from src.VStar import VectorizedVStar, compute_all_v_values
from src.likelihood import LikelihoodConfig, log_likelihood_optimized, calculate_weights

# Constants


class TestNetworkLikelihood(unittest.TestCase):
    def setUp(self):
        """Set up small test network with 3 nodes and 2 edges"""
        # Node attributes - 2D binary vectors
        self.node_attrs = {
            1: [1, 0],  # Node 1 attributes
            2: [0, 1],  # Node 2 attributes
            3: [1, 1]  # Node 3 attributes
        }

        # Network structure: 1-2-3 (two edges: 1-2 and 2-3)
        self.edges = {
            1: {2},
            2: {1, 3},
            3: {2}
        }

        # Node statistics (degree)
        self.node_stats = {
            1: 1,  # One connection
            2: 2,  # Two connections
            3: 1  # One connection
        }

        # Calculate weights
        self.weights = {
            node_id: calculate_weights(stats, self.node_stats.values())
            for node_id, stats in self.node_stats.items()
        }

        # Create likelihood config
        self.config = LikelihoodConfig(
            node_attrs=self.node_attrs,
            node_stats=self.node_stats,
            weights=self.weights,
            edges=self.edges
        )

        # Theta parameters for U* = β1*xi + β2*xj + β3*si + β4*sj
        # Where β1 and β2 are each 2-dimensional for the 2D attribute vectors
        self.theta = tf.constant([
            0.5, -0.5,  # β1: coefficients for xi
            0.3, -0.3,  # β2: coefficients for xj
            0.2,  # β3: coefficient for si (degree)
            -0.2  # β4: coefficient for sj (degree)
        ], dtype=tf.float16)

    def test_v_star_individual(self):
        """Test V* calculation for individual node pairs"""
        # Arrange
        v_star = VectorizedVStar()

        # Test for edge 1-2
        xi = tf.constant([self.node_attrs[1]], dtype=tf.float16)
        xj = tf.constant([self.node_attrs[2]], dtype=tf.float16)
        si = tf.constant([self.node_stats[1]], dtype=tf.float16)
        sj = tf.constant([self.node_stats[2]], dtype=tf.float16)

        # Act
        result = v_star(xi, xj, si, sj, self.theta)

        # Assert
        # Manual calculation:
        # U12 = 0.5*1 + (-0.5)*0 + 0.3*0 + (-0.3)*1 + 0.2*1 +(-0.2)*2 = 0.5 - 0.3 + 0.2 - 0.4 = 0
        # U21 = 0.5*0 + (-0.5)*1 + 0.3*1 + (-0.3)*0 + 0.2*2 +(-0.2)*1 = -0.5 + 0.3 + 0.4 - 0.2 = 0
        # V* = U12 + U21 = 0
        expected = tf.constant([[0.0]], dtype=tf.float16)

        self.assertTrue(tf.reduce_all(tf.abs(result - expected) < 1e-4))

    def test_v_star_batch(self):
        """Test V* calculation for batched node pairs"""
        # Arrange
        v_star = VectorizedVStar()

        # Create batch inputs
        xi = tf.constant([self.node_attrs[1], self.node_attrs[2]], dtype=tf.float16)[:, tf.newaxis, :]
        xj = tf.constant([self.node_attrs[2], self.node_attrs[3]], dtype=tf.float16)[:, tf.newaxis, :]
        si = tf.constant([[self.node_stats[1]], [self.node_stats[2]]], dtype=tf.float16)
        sj = tf.constant([[self.node_stats[2]], [self.node_stats[3]]], dtype=tf.float16)

        # Act
        result = v_star(xi, xj, si, sj, self.theta)

        # Assert
        # Shape should be [2, 2] for pairwise combinations
        self.assertEqual(result.shape, (2, 2))

        # Calculate expected values for specific pairs
        # For pair (1,2):
        u12 = 0.5*1 + (-0.5)*0 + 0.3*0 + (-0.3)*1 + 0.2*1 + (-0.2)*2  # = 0
        u21 = 0.5*0 + (-0.5)*1 + 0.3*1 + (-0.3)*0 + 0.2*2 + (-0.2)*1  # = -0.5 + 0.3 + 0.4 - 0.2 = 0
        expected_12 = u12 + u21  # = 0.0

        # For pair (1,3):
        u13 = 0.5 * 1 + (-0.5) * 0 + 0.3 * 1 + (-0.3) * 1 + 0.2 * 1 + (-0.2) * 1  # = 0.3
        u31 = 0.5 * 1 + (-0.5) * 1 + 0.3 * 1 + (-0.3) * 0 + 0.2 * 1 + (-0.2) * 1  # = 0.5
        expected_13 = u13 + u31  # = 0.8

        # For pair (2,3):
        u23 = 0.5*0 + (-0.5)*1 + 0.3*1 + (-0.3)*1 + 0.2*2 + (-0.2)*1  # = -0.3
        u32 = 0.5*1 + (-0.5)*1 + 0.3*0 + (-0.3)*1 + 0.2*1 + (-0.2)*2  # = -0.5
        expected_23 = u23 + u32  # = -0.8

        # For pair (2,2):
        u22 = 0.5 * 0 + (-0.5) * 1 + 0.3 * 0 + (-0.3) * 1 + 0.2 * 2 + (-0.2) * 2  # = -0.8
        expected_22 = u22*2  # = -1.6

        expected = tf.constant([
            [expected_12, expected_13],  # Node 1 paired with 2,3
            [expected_22, expected_23]  # Node 2 paired with 2,3
        ], dtype=tf.float16)

        self.assertTrue(tf.reduce_all(tf.abs(result - expected) < 1e-4))

    def test_psi_calculation(self):
        """Test PSI value calculation"""
        # Arrange
        x_tensor = tf.constant([self.node_attrs[1], self.node_attrs[2], self.node_attrs[3]], dtype=tf.float16)
        w_tensor = tf.constant([0.3, 0.3, 0.3], dtype=tf.float16)
        s_tensor = tf.constant([self.node_stats[1], self.node_stats[2], self.node_stats[3]], dtype=tf.float16)

        # Mock H function
        mock_h = Mock()
        mock_h.k_dim = 1
        mock_h.return_value = tf.ones([3, 1], dtype=tf.float16) * 0.5

        # Pre-compute V values
        v_values = compute_all_v_values(x_tensor, x_tensor,
                                        s_tensor, s_tensor,
                                        self.theta)

        # Act
        psi_values = compute_psi_values(x_tensor, x_tensor, w_tensor,
                                        s_tensor, mock_h, v_values)

        # Assert
        self.assertEqual(psi_values.shape, (3, 3, 1))
        # Manual calculation for each element
        h_value = 0.5
        w = 0.3

        # Calculate expected PSI values manually
        expected_psi = np.zeros((3, 3, 1))
        for i in range(3):
            for j in range(3):
                # Get V value for this pair
                v = v_values[i, j].numpy()
                # Get s value for node j
                s = s_tensor[j].numpy()
                # Calculate ψ = w * s * exp(V) / (1 + H)
                expected_psi[i, j, 0] = w * s * np.exp(v) / (1 + h_value)

        expected_psi = tf.constant(expected_psi, dtype=tf.float16)

        # Compare actual vs expected values
        np.testing.assert_array_almost_equal(psi_values.numpy(), expected_psi.numpy(), decimal=4)

    @patch('src.likelihood.H_star')
    def test_log_likelihood(self, mock_h_star):
        """Test log likelihood calculation with mocked H_star"""
        # Arrange
        # Create mock H* values
        mock_h = Mock()
        mock_h.k_dim = 1
        mock_h.return_value = tf.ones([4, 1], dtype=tf.float16) * 0.5
        mock_h_star.return_value = mock_h

        # Act
        ll, _ = log_likelihood_optimized(self.theta, self.config, H=mock_h)

        # Assert
        # Manual calculation for simple network:

        # For edge 1-2: L12=1, V12
        u12 = 0.5 * 1 + (-0.5) * 0 + 0.3 * 0 + (-0.3) * 1 + 0.2 * 1 + (-0.2) * 2  # = 0
        u21 = 0.5 * 0 + (-0.5) * 1 + 0.3 * 1 + (-0.3) * 0 + 0.2 * 2 + (-0.2) * 1  # = -0.5 + 0.3 + 0.4 - 0.2 = 0
        v12 = u12 + u21  # = 0.0
        # For edge 2-1: L21=1, V21
        v21 = u21 + u12  # = 0.0
        # For edge 2-3: L23=1, V23
        u23 = 0.5 * 0 + (-0.5) * 1 + 0.3 * 1 + (-0.3) * 1 + 0.2 * 2 + (-0.2) * 1  # = -0.3
        u32 = 0.5 * 1 + (-0.5) * 1 + 0.3 * 0 + (-0.3) * 1 + 0.2 * 1 + (-0.2) * 2  # = -0.5
        v23 = u23 + u32  # = -0.8
        # For edge 3-2: L32=1, V32
        v32 = u32 + u23
        # ll_edge = 0.5 * Lij * (Vij - log(1+Hi) - log(1+Hj))
        edge_ll = 0.5 * 1 * ((v12 - np.log(1.5) - np.log1p(.5)) + (v23 - np.log1p(.5) - np.log1p(.5)) +
                             (v21 - np.log(1.5) - np.log1p(.5)) + (v32 - np.log1p(.5) - np.log1p(.5)))
        # Node terms: log(s) - log(1 + H(x_i))
        node_ll = ((np.log(1) - np.log(1 + 0.5)) +  # Node 1 -> 2
                   (np.log(2) - np.log(1 + 0.5)) +  # Node 2 -> 1
                   (np.log(2) - np.log(1 + 0.5)) +  # Node 2 -> 3
                   (np.log(1) - np.log(1 + 0.5)))  # Node 3
        expected_ll = (edge_ll / 4) + (node_ll / 4)
        self.assertAlmostEqual(ll.numpy(), expected_ll, places=3)


if __name__ == '__main__':
    unittest.main()
