"""
-------------------------------------------------------
This module implements Hamiltonian Monte Carlo (HMC) sampling methods for
estimating parameters in network formation models. Provides both HMC sampling
and maximum likelihood estimation options.
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
import gc
import os
import tensorflow as tf
import tensorflow_probability as tfp
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


class MyHMCMC:
    """
    -------------------------------------------------------
    Hamiltonian Monte Carlo sampling and optimization for network formation models.
    Supports both HMC sampling and maximum likelihood parameter estimation.
    -------------------------------------------------------
    Parameters:
       num_dims_theta - dimension of parameter vector theta (int > 0)
       num_dims_h - dimension of H function output (int > 0, default=1)
       num_chains - number of parallel MCMC chains (int > 0, default=2)
       verbosity - whether to print detailed progress (bool, default=True)
       lr - learning rate for optimization (float > 0, default=1e-4)
       clip_val - gradient clipping threshold (float > 0, default=5)
    -------------------------------------------------------
    """

    def __init__(self, num_dims_theta: int, num_dims_h: int = 1, num_chains: int = 2, verbosity: bool = True,
                 lr: float = 1e-4, clip_val: float = 5):
        assert num_dims_theta > 0 and isinstance(num_dims_theta, int), "num_dims_theta must be a positive integer"
        assert num_dims_h > 0 and isinstance(num_dims_h, int), "num_dims_h must be a positive integer"
        assert num_chains > 0 and isinstance(num_chains, int), "num_chains must be a positive integer"
        assert lr > 0, "learning rate must be a positive float"
        assert clip_val > 0, "clip_val must be a positive float"

        self.num_dims_h = num_dims_h
        self.num_chains = num_chains
        self.verbosity = verbosity
        self.num_dims_theta = num_dims_theta
        self.param_state = None
        self.current_H = None
        self.current_ll = None
        self.learning_rate = lr
        self.clip_val = clip_val

    def log_likelihood_wrapper(self, likelihood_config: LikelihoodConfig) -> callable:
        """
        -------------------------------------------------------
        Creates a wrapper function for computing log likelihood with gradient tracking
        -------------------------------------------------------
        Parameters:
           likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
        Returns:
           log_prob - wrapped log likelihood function with gradient tracking (callable)
        -------------------------------------------------------
        """

        @tf.function
        def log_prob(theta):
            with tf.GradientTape() as tape:
                tape.watch(theta)
                # theta = tf.clip_by_value(theta, -5.0, 5.0)
                ll, H = log_likelihood_optimized(theta, likelihood_config, H=self.current_H)

            # Adding gradient clipping
            grads = tape.gradient(ll, theta)
            clipped_grads = tf.clip_by_value([grads], -self.clip_val, self.clip_val)[0]

            # Update H and likelihood
            self.current_H = H
            self.current_ll = ll
            log.debug(f"Log Likelihood: {ll}")
            log.debug(f"H: {repr(H)}")
            return ll / 1000  # likelihood is still too high normalize to allow model to explore its parameter space

        return log_prob

    def run_chain(self, likelihood_config: LikelihoodConfig, burn_in_steps=100, num_results=20):
        """
        -------------------------------------------------------
        Runs Hamiltonian Monte Carlo sampling chain
        -------------------------------------------------------
        Parameters:
           likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
           burn_in_steps - number of burn-in steps (int > 0, default=100)
           num_results - number of samples to collect (int > 0, default=20)
        Returns:
           samples - collected parameter samples from HMC chain (tf.Tensor)
        -------------------------------------------------------
        """
        assert self.param_state is not None, 'Initialize parameters first'

        tf.keras.backend.clear_session()

        # Define HMCMC kernel
        step_size = tf.fill([self.num_dims_theta], self.learning_rate)  # [0.1, 0.1, ...]
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_likelihood_wrapper(likelihood_config),
                num_leapfrog_steps=5,
                step_size=step_size),
            num_adaptation_steps=int(burn_in_steps * 0.8))

        # Run the chain
        # samples, [final_kernel_results] = tfp.mcmc.sample_chain(
        samples = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=burn_in_steps,
            current_state=self.param_state,
            kernel=adaptive_hmc,
            trace_fn=None,  # ,lambda _, pkr: [pkr],
            return_final_kernel_results=False)

        return samples  # , final_kernel_results

    def optimize(self, likelihood_config: LikelihoodConfig, burn_in_steps=100, num_results=20):
        """
        -------------------------------------------------------
        Optimizes parameters using HMC sampling
        -------------------------------------------------------
        Parameters:
           likelihood_config - configuration for likelihood computation (LikelihoodConfig)
           burn_in_steps - number of burn-in steps (int > 0, default=100)
           num_results - number of samples to collect (int > 0, default=20)
        Returns:
           best_sample - parameter values with highest likelihood (tf.Tensor)
           samples - all collected parameter samples (tf.Tensor)
           log_probs - log probabilities for all samples (list of tf.Tensor)
        -------------------------------------------------------
        """
        assert burn_in_steps > 0 and isinstance(burn_in_steps, int), "burn_in_steps must be a positive integer"
        assert num_results > 0 and isinstance(num_results, int), "num_results must be a positive integer"

        log.info("Optimizing Theta with HMC")
        # Initialize parameters
        if self.param_state is None:
            self.param_state = tf.abs(parameter_initializer([self.num_dims_theta], dtype=tf.float32))

        # Run the HMC Chain
        # samples, final_kernel_results = self.run_chain(node_attrs, node_stats, weights, burn_in_steps, num_results)
        samples = self.run_chain(likelihood_config, burn_in_steps, num_results)
        # log probability for sampled params
        log_probs = []
        for theta_sample in samples:
            ll, _ = log_likelihood_optimized(
                theta_sample, likelihood_config, H=self.current_H
            )
            log_probs.append(ll)

        if self.verbosity:
            log.info(f"Log Likelihood: {log_probs}")
            log.debug(f"H: {repr(self.current_H)}")
            # print(f"Theta: {samples}")
        # # Find the best sample
        best_idx = tf.argmax(log_probs)
        best_sample = samples[best_idx]

        return best_sample, samples, log_probs
        # return samples, log_probs

    def optimize_w_mle(self, likelihood_config: LikelihoodConfig, num_epochs):
        """
        -------------------------------------------------------
        Optimizes parameters using maximum likelihood estimation
        -------------------------------------------------------
        Parameters:
          likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
          num_epochs - number of training epochs (int > 0)
        Returns:
          losses - list of log likelihood values during training (list of tf.Tensor)
        -------------------------------------------------------
        """
        assert num_epochs > 0, "num_epochs must be a positive integer"

        log.info("Optimizing with MLE")
        self.param_state = tf.Variable(parameter_initializer([self.num_dims_theta], dtype=tf.float32))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        ll_function = self.log_likelihood_wrapper(likelihood_config)
        losses = []

        def train_step():
            with tf.GradientTape() as tape:
                ll = ll_function(self.param_state)

            # Get gradients
            grads = tape.gradient(ll, [self.param_state])
            # Clip gradients to avoid exploding gradients
            clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in grads]
            log.debug(f"{clipped_grads=}, {grads=}")
            # update parameters by applying gradients
            optimizer.apply_gradients(zip(clipped_grads, [self.param_state]))

            return ll

        for epoch in range(num_epochs):
            ll = train_step()
            gc.collect()
            losses.append(ll)
            log.info(f"Epoch {epoch + 1}, Log Likelihood: {-1 * ll}")

        return losses
