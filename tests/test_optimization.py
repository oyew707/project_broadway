"""
-------------------------------------------------------
Unit tests for optimization algorithms using a mock
likelihood function sum(-(4x)^2)
-------------------------------------------------------
Author: einsteinoyewole
Email: eo2233@nyu.edu
__updated__ = "12/15/24"
-------------------------------------------------------
"""

# Imports
import os
import sys
import shutil
import unittest
from unittest.mock import patch
import tensorflow as tf
from copy import deepcopy
from src.H import VectorizedH
from src.HMCMC import MyHMCMC
from src.reparameterizationVI import ReparameterizationVI
from src.likelihood import LikelihoodConfig
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)

class MockDataConfig:
    """Mock data configuration for testing"""

    def __init__(self, num_nodes=10):
        # Create mock node attributes and statistics
        self.node_attrs = {i: [1.0] for i in range(num_nodes)}
        self.node_stats = {i: 1 for i in range(num_nodes)}
        self.weights = {i: 1.0 for i in range(num_nodes)}
        self.edges = {i: set(range(num_nodes)) - {i} for i in range(num_nodes)}


@tf.function
def mock_log_likelihood(theta, likelihood_config, H=None, use_mean=False):
    """Mock likelihood function returning sum(-theta^2)"""
    theta = tf.cast(theta, tf.float32)
    ll = -tf.reduce_sum(tf.square(4*theta))

    if H is None:
        H = VectorizedH(k_dim=1,
                        node_attrs=likelihood_config.node_attrs,
                        node_stats=likelihood_config.node_stats)

    return ll, H


class TestOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods"""
        # set test directory
        cls.test_dir = "test_results"
        cls.best_model_dir = os.path.join(cls.test_dir, "best_model")

        # Initialize mock data config
        cls.config = MockDataConfig()
        cls.likelihood_config = LikelihoodConfig(
            node_attrs=cls.config.node_attrs,
            node_stats=cls.config.node_stats,
            weights=cls.config.weights,
            edges=cls.config.edges
        )

    def setUp(self):
        """Set up test fixtures specific to each test method"""
        # Clear any existing TF graph state
        tf.keras.backend.clear_session()
        # Clean test directories before each test
        if os.path.exists(self.best_model_dir):
            shutil.rmtree(self.best_model_dir)
        os.makedirs(self.best_model_dir)

    def tearDown(self):
        """Clean up after each test method"""
        # Clear TF session to free memory
        tf.keras.backend.clear_session()

    @patch('src.HMCMC.log_likelihood_optimized', side_effect=mock_log_likelihood)
    def test_hmcmc(self, mock_likelihood):
        """Test HMCMC optimization"""
        # Arrange
        num_dims_theta = 3
        hmcmc = MyHMCMC(
            num_dims_theta=num_dims_theta,
            num_chains=5,  # More chains
            verbosity=True,
            lr=0.0001,  # Smaller learning rate
            leap_frog=50,  # More leapfrog steps
            clip_val=1.0  # Tighter gradient clipping
        )
        init_state = tf.random.normal([num_dims_theta], mean=0.0, stddev=0.001)
        hmcmc.param_state = tf.Variable(init_state)

        # Act
        best_sample, samples, log_probs = hmcmc.optimize(
            likelihood_config=self.likelihood_config,
            burn_in_steps=2500,
            num_results=200,
        )

        # assert
        self.assertTrue(mock_likelihood.called)
        likelihood = mock_log_likelihood(best_sample, self.likelihood_config)[0]
        self.assertLess(abs(float(likelihood)), 0.1,
                        f"Likelihood not close enough to optimal: {likelihood}, params={best_sample}")

    @patch('src.HMCMC.log_likelihood_optimized', side_effect=mock_log_likelihood)
    def test_mle(self, mock_likelihood):
        """Test MLE optimization"""
        # Arrange
        hmcmc = MyHMCMC(
            num_dims_theta=3,
            verbosity=True,
            lr=0.001,
            clip_val=5.0
        )
        init_state = tf.random.normal([3], mean=0.0, stddev=0.001)
        hmcmc.param_state = tf.Variable(init_state)

        # Act
        _ = hmcmc.optimize_w_mle(
            likelihood_config=self.likelihood_config,
            run_path=self.test_dir,
            num_epochs=100
        )

        # Assert
        self.assertTrue(mock_likelihood.called)
        loaded_hmcmc = deepcopy(hmcmc)
        loaded_hmcmc.load_state(self.best_model_dir, self.likelihood_config)
        params = loaded_hmcmc.param_state
        log_likelihood = mock_log_likelihood(params, self.likelihood_config)[0]
        self.assertAlmostEqual(float(log_likelihood), 0.0, 3, f"MLE likelihood not close to optimal: {params=}")

    @patch('src.reparameterizationVI.log_likelihood_optimized', side_effect=mock_log_likelihood)
    def test_vi(self, mock_likelihood):
        """Test Variational Inference optimization"""
        # Arrange
        vi = ReparameterizationVI(
            num_dims_theta=3,
            verbose=True,
            lr=1e-3,
            clip_val=10.0,
            num_epochs=300,
            num_samples=1
        )
        init_state = tf.random.normal([3], mean=0.0, stddev=0.001)
        vi.variational_params = {
            'mean': tf.Variable(init_state),
            'variance': tf.Variable(tf.ones_like(init_state)*0.5)
        }

        # Act
        _ = vi.optimize(
            likelihood_config=self.likelihood_config,
            run_path=self.test_dir,
            optimize_alpha=True,
            patience=20
        )

        # Assert
        self.assertTrue(mock_likelihood.called)
        new_vi = deepcopy(vi)
        new_vi.load_state(self.best_model_dir, self.likelihood_config)
        params = new_vi.variational_params['mean']
        log_likelihood = mock_log_likelihood(params, self.likelihood_config)[0]
        self.assertLess(abs(float(log_likelihood)), 0.01,
                        f"VI likelihood not close enough to optimal: {log_likelihood}, params={params}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        # Clean up test directories
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
