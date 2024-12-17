"""
-------------------------------------------------------
Implements the reparameterization variational inference algorithm
for network formation models using Normal distribution for variational
parameters and Gamma distribution for the conjugate prior.
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "12/14/24"
-------------------------------------------------------
"""

# Imports
import os
import pickle
from typing import Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from src.H import VectorizedH
from src.likelihood import log_likelihood_optimized, LikelihoodConfig
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)
tf.config.run_functions_eagerly(True)
SEED = int(os.getenv('RANDOMSEED', '4'))
tf.random.set_seed(SEED)
mean_parameter_initializer = tf.keras.initializers.RandomNormal(
    seed=SEED
)
var_parameter_initializer = tf.keras.initializers.RandomUniform(
    minval=0.1,  # Small positive value
    maxval=1.0,
    seed=SEED
)


class ReparameterizationVI:
    """
    -------------------------------------------------------
    Implements the reparameterization variational inference algorithm
    for network formation models.
    -------------------------------------------------------
    Parameters:
        verbose - whether to print progress messages (bool)
        num_dims_theta - number of dimensions for the model parameters (int)
        lr - learning rate for optimization (float)
        num_epochs - number of training epochs (int)
        num_samples - number of samples for the variational distribution (int)
        clip_val - value for gradient clipping (float)
        num_dims_h - number of dimensions for the H values (int)
    -------------------------------------------------------
    """

    def __init__(self, verbose: bool = True, num_dims_theta: int = 10, lr: float = 0.01, num_epochs: int = 10,
                 num_samples: int = 1, clip_val: float = 5, num_dims_h: int = 1):
        assert num_dims_theta > 0 and isinstance(num_dims_theta,
                                                 int), "Number of dimensions for model parameters must be greater than 0"
        assert lr > 0 and isinstance(lr, float), "Learning rate must be greater than 0"
        assert num_epochs > 0 and isinstance(num_epochs, int), "Number of epochs must be greater than 0"
        assert num_samples > 0 and isinstance(num_samples, int), "Number of samples must be greater than 0"
        # Use only one sample for optimization, because H is computed through an iterative fixed-point computation
        # Each sample of θ would require a different fixed point for H. this computation is expensive
        # and without clear benefits for convergence.
        assert num_samples == 1, "Number of samples must be 1"

        self.verbose = verbose
        self.num_dims_theta = num_dims_theta
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.variational_params = None
        self.conjugate_prior_parameters = None
        self.param_state = None
        self.current_H = None
        self.current_ll = None
        self.clip_val = clip_val
        self.num_dims_h = num_dims_h

    def log_likelihood_wrapper(self, likelihood_config: LikelihoodConfig, use_mean: bool = False) -> callable:
        """
        -------------------------------------------------------
        Creates a wrapper function for computing log likelihood with gradient tracking
        -------------------------------------------------------
        Parameters:
           likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
              use_mean - whether to use the mean for normalizing likelihood (bool, default=False
        Returns:
           log_prob - wrapped log likelihood function with gradient tracking (callable)
        -------------------------------------------------------
        """

        @tf.function
        def log_prob(theta):
            with tf.GradientTape() as tape:
                tape.watch(theta)
                # theta = tf.clip_by_value(theta, -5.0, 5.0)
                ll, H = log_likelihood_optimized(theta, likelihood_config, H=self.current_H, use_mean=use_mean)

            # Update H and likelihood
            self.current_H = H
            self.current_ll = ll
            if self.verbose:
                log.debug(f"Log Likelihood: {ll}")
                log.debug(f"Theta: {theta}")
                log.debug(f"H: {repr(H)}")
            return ll

        return log_prob

    def _initialize_parameters(self):
        """
        -------------------------------------------------------
        Initializes the model parameters and variational distribution
        parameters
        -------------------------------------------------------
        """
        if self.variational_params is None:
            self.variational_params = {
                'mean': tf.Variable(
                    initial_value=mean_parameter_initializer(shape=[self.num_dims_theta], dtype=tf.float32),
                    trainable=True
                ),
                'variance': tf.Variable(
                    initial_value=tf.abs(var_parameter_initializer(shape=[self.num_dims_theta], dtype=tf.float32)),
                    dtype=tf.float32,
                    name='variance',
                    trainable=True
                )
            }
        if self.conjugate_prior_parameters is None:
            self.conjugate_prior_parameters = {
                'alpha': tf.Variable(
                    initial_value=2*tf.ones(shape=[self.num_dims_theta], dtype=tf.float32),
                    trainable=True
                ),
                'beta': tf.Variable(
                    initial_value=tf.ones(shape=[self.num_dims_theta], dtype=tf.float32),
                    trainable=True
                )
            }

    def variational_distribution(self, **kwargs):
        """
        -------------------------------------------------------
        Creates the variational distribution for the model parameters, q(θ)
        -------------------------------------------------------
        Returns:
            distribution - variational distribution for the model parameters (tfp.distributions.Distribution)
        """
        assert self.variational_params is not None, "Variational parameters must be initialized"
        if kwargs.get('alpha') is not None:
            assert self.conjugate_prior_parameters is not None, "Variational parameters must be initialized"
            return tfp.distributions.Gamma(
                concentration=self.conjugate_prior_parameters['alpha'],
                rate=self.conjugate_prior_parameters['beta']
            )
        var = tf.nn.softplus(self.variational_params['variance'])
        return tfp.distributions.MultivariateNormalDiag(
            loc=self.variational_params['mean'],
            scale_diag=tf.math.sqrt(var)
        )

    def sample_theta(self, alpha: Optional[tf.Tensor] = None):
        """
        -------------------------------------------------------
        Samples model parameters from the variational distribution
        using the reparameterization trick. If alpha is provided,
        the samples are conditioned on alpha
        -------------------------------------------------------
        Parameters:
            alpha - hyperparameters for the conjugate prior (tf.Tensor, optional)
        Returns:
            theta - sampled model parameters (tf.Tensor)
        """
        assert self.variational_params is not None, "Variational parameters must be initialized"
        epsilon = tf.random.normal(shape=[self.num_dims_theta], dtype=tf.float32, )
        # Helps prevent negative values for variance
        var = tf.nn.softplus(self.variational_params['variance'])
        mu = self.variational_params['mean']
        if alpha is None:
            log.debug(f"{mu=}, {var=}, {epsilon=}")
            theta = mu + epsilon * tf.math.sqrt(var)
        else:
            log.debug(f"{mu=}, {var=}, {epsilon=} {alpha=}")
            theta = mu + tf.sqrt(1 / alpha) * epsilon * tf.math.sqrt(var)
        return theta

    def sample_alpha(self):
        """
        -------------------------------------------------------
        Samples the hyperparameters for the conjugate prior
        -------------------------------------------------------
        Returns:
            alpha - sampled hyperparameters (tf.Tensor)
        """
        assert self.conjugate_prior_parameters is not None, "Conjugate prior parameters must be initialized"
        alpha = tfp.distributions.Gamma(self.conjugate_prior_parameters['alpha'],
                                        self.conjugate_prior_parameters['beta']).sample()
        return tf.maximum(alpha, 0.01)

    def log_prior_theta(self, theta: tf.Tensor, alpha: Optional[tf.Tensor]):
        """
        -------------------------------------------------------
        Computes the log prior for the model parameters
        p(θ|α) = N(θ|0, α⁻¹)  # Prior on parameters
        p(α) = Gamma(α|a,b)    # Hyperprior on precision
        p(θ,α) = p(θ|α)p(α)
        -------------------------------------------------------
        Parameters:
            theta - model parameters (tf.Tensor)
            alpha - conjugate prior hyperparameters (tf.Tensor, optional)
        Returns:
            log_prior - computed log prior (tf.Tensor)
        """
        assert self.conjugate_prior_parameters is not None, "Conjugate prior parameters must be initialized"
        if alpha is None:
            # Integrate out α analytically Results in Student-t distribution
            return tf.reduce_sum(
                tfp.distributions.StudentT(df=2 * self.conjugate_prior_parameters['alpha'], loc=0., scale=tf.sqrt(
                    self.conjugate_prior_parameters['beta'] / self.conjugate_prior_parameters['alpha'])).log_prob(theta)
            )
        # Normal prior on theta
        log_p_theta = tf.reduce_sum(
            tfp.distributions.Normal(0., tf.math.sqrt(1 / alpha)).log_prob(theta)
        )
        # Gamma prior on alpha
        log_p_alpha = tfp.distributions.Gamma(self.conjugate_prior_parameters['alpha'],
                                              self.conjugate_prior_parameters['beta']).log_prob(alpha)
        return log_p_theta + log_p_alpha

    @tf.function
    def compute_elbo(self, log_likelihood: Callable, optimize_alpha: bool = False):
        """
        -------------------------------------------------------
        Computes the evidence lower bound (ELBO) for the model
        -------------------------------------------------------
        Parameters:
            likelihood_config - configuration for likelihood computation (LikelihoodConfig)
            optimize_alpha - whether to optimize the conjugate hyperparameters (bool)
        Returns:
            elbo - computed evidence lower bound (tf.Tensor)
        ----------------------------------------------------
        """
        # Draws 1 sample from alpha
        alpha = self.sample_alpha() if optimize_alpha else None
        # Draws 1 sample from theta
        theta = self.sample_theta(alpha=alpha)
        log.debug(f"{theta=}, {alpha=}")
        # Compute log q; q(θ,α;λ) = q(θ|α;λ_θ)q(α;λ_α)
        log_q = (tf.reduce_sum(self.variational_distribution().log_prob(theta)) +
                 tf.reduce_sum(self.variational_distribution(alpha=True).log_prob(alpha))) if optimize_alpha else (
                    tf.reduce_sum(self.variational_distribution().log_prob(theta)))
        # Compute log likelihood
        log_likelihood= log_likelihood(theta)
        log.debug(f'{log_likelihood=}, {log_q=}')
        # Compute joint probability
        log_p_x_theta = log_likelihood + self.log_prior_theta(theta, None)

        # Compute ELBO
        elbo = log_p_x_theta - log_q
        log.debug(f"{elbo=}")

        # set theta as param_state
        self.param_state = theta  # if sample > 1, should be average
        return elbo

    def optimize(self, likelihood_config: LikelihoodConfig, run_path: str,
                 optimize_alpha: bool = False, patience: int = 3):
        """
        -------------------------------------------------------
        Optimizes the model parameters using the reparameterization
        variational inference algorithm
        -------------------------------------------------------
        Parameters:
            likelihood_config - configuration for likelihood computation (LikelihoodConfig)
            run_path - directory to save state files (str)
            optimize_alpha - whether to optimize the conjugate hyperparameters (bool)
            patience - number of epochs to wait for improvement before stopping (int)
        Returns:
            losses - list of computed losses for each epoch (list)
        -------------------------------------------------------
        """
        assert self.num_epochs >= patience > 0, "Patience must be greater than 0 and less than the number of epochs"

        losses = []
        prev_loss = float('inf')
        best_loss = float('inf')
        not_improve_count = 0

        # Initialize parameters
        self._initialize_parameters()
        log_prob = self.log_likelihood_wrapper(likelihood_config, use_mean=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.num_epochs):
            with tf.GradientTape() as tape:
                elbo = self.compute_elbo(log_prob, optimize_alpha)
                loss = -elbo
            losses.append(loss)

            # Save best model
            if loss < best_loss:
                best_loss = loss
                new_path = os.path.join(run_path, 'best_model')
                self.save_state(new_path)

            if optimize_alpha:
                gradients = tape.gradient(loss, [self.variational_params['mean'],
                                                 self.variational_params['variance'],
                                                 self.conjugate_prior_parameters['alpha'],
                                                 self.conjugate_prior_parameters['beta']])
                clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in gradients]
                log.debug(f"{clipped_grads=}, {gradients=}")
                optimizer.apply_gradients(
                    zip(gradients, [self.variational_params['mean'],
                                    self.variational_params['variance'],
                                    self.conjugate_prior_parameters['alpha'],
                                    self.conjugate_prior_parameters['beta']])
                )
            else:
                gradients = tape.gradient(loss, [self.variational_params['mean'], self.variational_params['variance']])
                clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in gradients]
                log.debug(f"{clipped_grads=}, {gradients=}")
                optimizer.apply_gradients(
                    zip(gradients, [self.variational_params['mean'], self.variational_params['variance']]))

            if prev_loss - loss < 1e-6:
                not_improve_count += 1
                if not_improve_count >= patience:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                not_improve_count = 0
            prev_loss = loss
            # Clear memory
            tf.keras.backend.clear_session()
            # Print progress
            if self.verbose:
                log.info(f"Epoch: {epoch}, ELBO: {elbo}")
        return losses

    def save_state(self, directory: str, **kwargs):
        """
        -------------------------------------------------------
        Saves the current state of the VI including H values
        and parameter state to files
        -------------------------------------------------------
        Parameters:
            directory - directory to save state files (str)
        -------------------------------------------------------
        """
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # Save H values if they exist
        if self.current_H is not None:
            h_path = os.path.join(directory, 'current_h.pkl')
            self.current_H.save(h_path)
            log.info(f'Saved current H values to {h_path}')

        # Save parameter state and other variables
        state_dict = {
            'variational_params': self.variational_params if self.variational_params is not None else None,
            'conjugate_prior_parameters': self.conjugate_prior_parameters if self.conjugate_prior_parameters is not None else None,
            'current_ll': self.current_ll,
            'num_dims_theta': self.num_dims_theta,
            'learning_rate': self.lr,
            'num_dims_h': self.num_dims_h,
            'clip_val': self.clip_val,
            'param_state': self.param_state
        }

        state_path = os.path.join(directory, 'vi_state.pkl')
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state_dict, f)
            log.info(f'Saved Variational Inference state to {state_path}')
        except Exception as e:
            log.error(f'Failed to save Variational Inference state to {state_path}: {str(e)}')
            raise

    def load_state(self, directory: str, likelihood_config: LikelihoodConfig, state_type: str = 'vi'):
        """
        -------------------------------------------------------
        Loads the VI state from files in the given directory
        -------------------------------------------------------
        Parameters:
            directory - directory containing state files (str)
            likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
            state_type - type of state to load (str, default='vi')
        -------------------------------------------------------
        """
        assert state_type in ['hmcmc', 'vi'], f"Invalid state type: {state_type}"
        directory = os.path.join(os.getcwd(), directory)

        file_name = 'hmcmc_state.pkl' if state_type == 'hmcmc' else 'vi_state.pkl'
        # Load parameter state and other variables
        state_path = os.path.join(directory, file_name)
        assert os.path.isfile(state_path), FileNotFoundError(f"No state file found at {state_path}")

        try:
            with open(state_path, 'rb') as f:
                state_dict = pickle.load(f)

            # Restore parameter state
            if state_dict.get('variational_params') is not None:
                self.variational_params = state_dict['variational_params']

            # Restore conjugate prior parameters
            if state_dict.get('conjugate_prior_parameters') is not None:
                self.conjugate_prior_parameters = state_dict['conjugate_prior_parameters']

            # Restore other variables
            self.current_ll = state_dict.get('current_ll', self.current_ll)
            self.num_dims_theta = state_dict.get('num_dims_theta', self.num_dims_theta)
            self.num_dims_h = state_dict.get('num_dims_h', self.num_dims_h)
            self.lr = state_dict.get('learning_rate', self.lr)
            self.clip_val = state_dict.get('clip_val', self.clip_val)
            self.param_state = state_dict.get('param_state', self.param_state)

            log.info(f'Successfully loaded Variational Inference state from {directory}')

        except Exception as e:
            log.error(f'Failed to load Variational Inference state from {directory}: {str(e)}')
            raise

        # Load H values if they exist
        h_path = os.path.join(directory, 'current_h.pkl')
        if os.path.isfile(h_path):
            if self.current_H is None:
                self.current_H = VectorizedH(k_dim=self.num_dims_h, node_attrs=likelihood_config.node_attrs,
                                             node_stats=likelihood_config.node_stats)
            self.current_H.load(h_path)
            log.info(f'Loaded H values from {h_path}')

        return
