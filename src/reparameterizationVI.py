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
from typing import Callable, Optional
import pickle
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
parameter_initializer = tf.keras.initializers.RandomNormal(
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
        self.variational_params = {
            'mean': tf.Variable(
                initial_value=parameter_initializer(shape=[self.num_dims_theta], dtype=tf.float32),
                trainable=True
            ),
            'variance': tfp.util.TransformedVariable(
                initial_value=parameter_initializer(shape=[self.num_dims_theta], dtype=tf.float32),
                bijector=tfp.bijectors.Softplus(),
                dtype=tf.float32,
                name='variance',
                trainable=True
            )
        }
        self.conjugate_prior_parameters = {
            'alpha': tf.Variable(
                initial_value=tf.ones(shape=[self.num_dims_theta], dtype=tf.float32),
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
        if kwargs['alpha'] is True:
            return tfp.distributions.Gamma(
                concentration=self.variational_params['alpha'],
                rate=self.variational_params['beta']
            )
        return tfp.distributions.Normal(
            loc=self.variational_params['mean'],
            scale=tf.math.sqrt(self.variational_params['variance'])
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
        epsilon = tf.random.normal(shape=[self.num_dims_theta, self.num_samples], dtype=tf.float32)
        if alpha is None:
            theta = self.variational_params['mean'] + epsilon * tf.math.sqrt(self.variational_params['variance'])
        else:
            theta = self.variational_params['mean'] + tf.sqrt(1 / alpha) * epsilon * tf.math.sqrt(
                self.variational_params['variance'])
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
        return alpha

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
        """
        # Draws 1 sample from alpha
        alpha = self.sample_alpha() if optimize_alpha else None
        # Draws 1 sample from theta
        theta = self.sample_theta(alpha=alpha)
        # Compute log q; q(θ,α;λ) = q(θ|α;λ_θ)q(α;λ_α)
        log_q = self.variational_distribution().log_prob(theta) + self.variational_distribution(alpha=True).log_prob(
            alpha) if optimize_alpha else self.variational_distribution().log_prob(theta)
        # Compute log likelihood
        log_likelihood = log_likelihood(theta)
        # Compute joint probability
        log_p_x_theta = log_likelihood + self.log_prior_theta(theta, None)

        # Compute ELBO
        elbo = log_p_x_theta - log_q
        return elbo

    def optimize(self, likelihood_config: LikelihoodConfig, optimize_alpha: bool = False):
        """
        -------------------------------------------------------
        Optimizes the model parameters using the reparameterization
        variational inference algorithm
        -------------------------------------------------------
        Parameters:
            likelihood_config - configuration for likelihood computation (LikelihoodConfig)
            optimize_alpha - whether to optimize the conjugate hyperparameters (bool)
        Returns:
            losses - list of computed losses for each epoch (list)
        -------------------------------------------------------
        """
        losses = []
        # Initialize parameters
        self._initialize_parameters()
        log_prob = self.log_likelihood_wrapper(likelihood_config, use_mean=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.num_epochs):
            with tf.GradientTape() as tape:
                elbo = self.compute_elbo(log_prob, optimize_alpha)
                loss = -elbo
            losses.append(loss)
            if optimize_alpha:
                gradients = tape.gradient(loss, [self.variational_params['mean'],
                                                 self.variational_params['variance'],
                                                 self.conjugate_prior_parameters['alpha'],
                                                 self.conjugate_prior_parameters['beta']])
                clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in grads]
                log.debug(f"{clipped_grads=}, {gradients=}")
                optimizer.apply_gradients(
                    zip(gradients, [self.variational_params['mean'],
                                    self.variational_params['variance'],
                                    self.conjugate_prior_parameters['alpha'],
                                    self.conjugate_prior_parameters['beta']])
                )
            else:
                gradients = tape.gradient(loss, [self.variational_params['mean'], self.variational_params['variance']])
                clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in grads]
                log.debug(f"{clipped_grads=}, {gradients=}")
                optimizer.apply_gradients(
                    zip(gradients, [self.variational_params['mean'], self.variational_params['variance']]))
            if self.verbose:
                log.info(f"Epoch: {epoch}, ELBO: {elbo}")
        return losses

    def save_state(self, directory: str):
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
            'clip_val': self.clip_val
        }

        state_path = os.path.join(directory, 'vi_state.pkl')
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state_dict, f)
            log.info(f'Saved Variational Inference state to {state_path}')
        except Exception as e:
            log.error(f'Failed to save Variational Inference state to {state_path}: {str(e)}')
            raise

    def load_state(self, directory: str, likelihood_config: LikelihoodConfig):
        """
        -------------------------------------------------------
        Loads the VI state from files in the given directory
        -------------------------------------------------------
        Parameters:
            directory - directory containing state files (str)
            likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
        -------------------------------------------------------
        """
        directory = os.path.join(os.getcwd(), directory)

        # Load parameter state and other variables
        state_path = os.path.join(directory, 'vi_state.pkl')
        assert os.path.isfile(state_path), FileNotFoundError(f"No state file found at {state_path}")

        try:
            with open(state_path, 'rb') as f:
                state_dict = pickle.load(f)

            # Restore parameter state
            if state_dict['variational_params'] is not None:
                self.variational_params = tf.Variable(state_dict['variational_params'])

            # Restore conjugate prior parameters
            if state_dict['conjugate_prior_parameters'] is not None:
                self.conjugate_prior_parameters = tf.Variable(state_dict['conjugate_prior_parameters'])

            # Restore other variables
            self.current_ll = state_dict['current_ll']
            self.num_dims_theta = state_dict['num_dims_theta']
            self.lr = state_dict['learning_rate']
            self.clip_val = state_dict['clip_val']

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

