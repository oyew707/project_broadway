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
import pickle
import random
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
random.seed(SEED)
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
       leap_frog - number of leap frog steps for HMC (int >= 0, default=2)
       verbosity - whether to print detailed progress (bool, default=True)
       lr - learning rate for optimization (float > 0, default=1e-4)
       clip_val - gradient clipping threshold (float > 0, default=5)
    -------------------------------------------------------
    """

    def __init__(self, num_dims_theta: int, num_dims_h: int = 1, num_chains: int = 2, leap_frog: int = 2,
                 verbosity: bool = True, lr: float = 1e-4, clip_val: float = 5):
        assert num_dims_theta > 0 and isinstance(num_dims_theta, int), "num_dims_theta must be a positive integer"
        assert num_dims_h > 0 and isinstance(num_dims_h, int), "num_dims_h must be a positive integer"
        assert num_chains > 0 and isinstance(num_chains, int), "num_chains must be a positive integer"
        assert leap_frog >= 0 and isinstance(leap_frog, int), "leap_frog must be a positive integer"
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
        self.leap_frog = leap_frog

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

            # Adding gradient clipping
            grads = tape.gradient(ll, theta)
            clipped_grads = tf.clip_by_value([grads], -self.clip_val, self.clip_val)[0]

            # Update H and likelihood
            self.current_H = H
            self.current_ll = ll
            if self.verbosity:
                log.debug(f"Log Likelihood: {ll}")
                log.debug(f"Theta: {theta}")
                log.debug(f"H: {repr(H)}")
            return ll

        return log_prob

    @tf.function
    def run_chain(self, likelihood_config: LikelihoodConfig, seed: int, burn_in_steps: int = 100,
                  num_results: int = 20):
        """
        -------------------------------------------------------
        Runs Hamiltonian Monte Carlo sampling chain
        -------------------------------------------------------
        Parameters:
           likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
           seed - random seed for reproducibility (int)
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
                target_log_prob_fn=self.log_likelihood_wrapper(likelihood_config, use_mean=True),
                num_leapfrog_steps=self.leap_frog,
                step_size=step_size,
                state_gradients_are_stopped=True
            ),
            num_adaptation_steps=int(burn_in_steps * 0.8))

        # Run the chain
        # samples, [final_kernel_results] = tfp.mcmc.sample_chain(
        samples = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=burn_in_steps,
            current_state=self.param_state,
            kernel=adaptive_hmc,
            parallel_iterations=1,
            trace_fn=None,  # ,lambda _, pkr: [pkr],
            return_final_kernel_results=False,
            seed=seed
        )

        # Compute Gelman-Rubin statistic
        r_hat = tfp.mcmc.potential_scale_reduction(tf.stack(samples))

        # Compute effective sample size
        ess = tfp.mcmc.effective_sample_size(tf.stack(samples))

        log.info(f"R-hat statistic: {r_hat}")
        log.info(f"Effective sample size: {ess}")

        # Consider chain converged if R-hat < 1.1 for all parameters
        is_converged = tf.reduce_all(r_hat < 1.1)
        log.info(f"Chain convergence after {burn_in_steps} steps: {is_converged}")
        return samples  # , final_kernel_results

    @tf.function
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
        samples = []
        chain_seed = random.choices(range(self.num_chains*100), k=self.num_chains)
        # samples, final_kernel_results = self.run_chain(node_attrs, node_stats, weights, burn_in_steps, num_results)
        for i in range(self.num_chains):
            sample = self.run_chain(likelihood_config, chain_seed[i], burn_in_steps, num_results // self.num_chains)
            samples.extend(sample.numpy().tolist())

            # Clear memory
            tf.keras.backend.clear_session()
        # log probability for sampled params
        log_probs = []
        for theta_sample in samples:
            ll, _ = log_likelihood_optimized(
                theta_sample, likelihood_config, H=self.current_H, use_mean=True
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

    def optimize_w_mle(self, likelihood_config: LikelihoodConfig, num_epochs: int, run_path: str, patience: int = 3):
        """
        -------------------------------------------------------
        Optimizes parameters using maximum likelihood estimation
        -------------------------------------------------------
        Parameters:
          likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
          run_path - directory to save state files (str)
          num_epochs - number of training epochs (int > 0)
            patience - number of epochs to wait before early stopping (int > 0, default=3)
        Returns:
          losses - list of log likelihood values during training (list of tf.Tensor)
        -------------------------------------------------------
        """
        assert num_epochs > 0, "num_epochs must be a positive integer"
        assert num_epochs >= patience >= 0, "patience must be a positive integer and less than epochs"

        log.info("Optimizing with MLE")
        self.param_state = tf.Variable(parameter_initializer([self.num_dims_theta], dtype=tf.float32),
                                       trainable=True) if self.param_state is None else self.param_state
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        ll_function = self.log_likelihood_wrapper(likelihood_config, use_mean=True)
        losses = []
        best_loss = float('inf')
        prev_loss = float('inf')
        improve_count = 0

        def train_step(best_loss=best_loss):
            with tf.GradientTape() as tape:
                ll = ll_function(self.param_state)
                ll *= -1  # Negative log likelihood

            # Save best parameters
            if ll < best_loss:
                new_path = os.path.join(run_path, 'best_model')
                self.save_state(new_path)

            # Get gradients
            grads = tape.gradient(ll, [self.param_state])
            # Clip gradients to avoid exploding gradients
            clipped_grads = [tf.clip_by_value(g, -self.clip_val, self.clip_val) for g in grads]
            log.debug(f"{clipped_grads=}, {grads=}")
            # update parameters by applying gradients
            old_params = self.param_state.numpy()
            optimizer.apply_gradients(zip(clipped_grads, [self.param_state]))
            new_params = self.param_state.numpy()
            log.debug(f"Parameter changes: {new_params - old_params}")

            return ll

        for epoch in range(num_epochs):
            ll = train_step(best_loss)

            # Save best loss
            best_loss = ll if ll < best_loss else best_loss

            # Early stopping
            if prev_loss - ll < 1e-6:
                improve_count += 1
                if improve_count >= patience:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                improve_count = 0
            prev_loss = ll
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()

            # Store loss
            losses.append(ll)
            log.info(f"Epoch {epoch + 1}, Log Likelihood: {-1 * ll}")
            log.debug(f"Theta: {self.param_state}")

        return losses

    def save_state(self, directory: str):
        """
        -------------------------------------------------------
        Saves the current state of the HMCMC sampler including H values
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
            'param_state': self.param_state.numpy() if self.param_state is not None else None,
            'current_ll': self.current_ll,
            'num_dims_theta': self.num_dims_theta,
            'num_dims_h': self.num_dims_h,
            'num_chains': self.num_chains,
            'learning_rate': self.learning_rate,
            'clip_val': self.clip_val,
            'leap_frog': self.leap_frog
        }

        state_path = os.path.join(directory, 'hmcmc_state.pkl')
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state_dict, f)
            log.info(f'Saved HMCMC state to {state_path}')
        except Exception as e:
            log.error(f'Failed to save HMCMC state to {state_path}: {str(e)}')
            raise

    def load_state(self, directory: str, likelihood_config: LikelihoodConfig, state_type: str = 'hmcmc'):
        """
        -------------------------------------------------------
        Loads the HMCMC state from files in the given directory
        -------------------------------------------------------
        Parameters:
            directory - directory containing state files (str)
            likelihood_config - data configuration for likelihood computation (LikelihoodConfig)
            state_type - type of state to load ('hmcmc' or 'vi', default='hmcmc')
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
            if state_dict.get('param_state') is not None:
                self.param_state = tf.Variable(state_dict['param_state'])

            # Restore other variables
            self.current_ll = state_dict.get('current_ll', self.current_ll)
            self.num_dims_theta = state_dict.get('num_dims_theta', self.num_dims_theta)
            self.num_dims_h = state_dict.get('num_dims_h', self.num_dims_h)
            self.num_chains = state_dict.get('num_chains', self.num_chains)
            self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
            self.clip_val = state_dict.get('clip_val', self.clip_val)
            self.leap_frog = state_dict.get('leap_frog', self.leap_frog)

            log.info(f'Successfully loaded HMCMC state from {directory}')

        except Exception as e:
            log.error(f'Failed to load HMCMC state from {directory}: {str(e)}')
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
