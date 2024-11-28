"""
-------------------------------------------------------
[Program Description]
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
from src.dataset import load_data, process_data
from src.likelihood import calculate_weights, LikelihoodConfig
from src.HMCMC import MyHMCMC
from src.logger import getlogger
import atexit

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


def main():
    # Load and Preprocess data
    node_df, edge_df, committee_df = load_data(data_path='./data')
    node_attrs, transactions, network_stats, edges = process_data(node_df, edge_df, committee_df)
    node_stats = {k: v['degree'] for k, v in network_stats.items()}

    # Calculate importance weights
    weights = {node_id: calculate_weights(s > 0, node_stats.values()) for node_id, s in node_stats.items()}

    # Likelihood Config
    config = LikelihoodConfig(node_attrs=node_attrs, node_stats=node_stats, edges=edges, weights=weights)

    # Define HMCMC parameters and optimizer object
    num_results = 20
    num_burnin_steps = 100
    x_dim = len(list(node_attrs.values())[0])
    s_dim = 1  # len(list(node_stats.values())[0])
    hmcmc_optimizer = MyHMCMC(
        num_dims_theta=2 * (x_dim + s_dim),  # Length of theta for x_i, x_j ,s_i and s_j used in computation of U*
        num_dims_h=1,  # Set to 1 because initial values are the average node stats
        num_chains=2,  # Number of different chains to run (should be concurrent)
        verbosity=True,
        lr=1e-4,  # Learning rate for optimization
        clip_val=5  # Gradient clipping threshold
    )
    # Save state at exit in case of crash
    atexit.register(hmcmc_optimizer.save_state, 'hmcmc_checkpoint_exit')

    # Optimize the likelihood function with MLE
    log.info("Optimizing likelihood function with MLE")
    losses = hmcmc_optimizer.optimize_w_mle(config, num_epochs=20)
    log.info(f"Log-likelihood: {losses[-1]}: MLE theta: {hmcmc_optimizer.param_state}")
    hmcmc_optimizer.save_state('hmcmc_checkpoint_mle')

    # Run HMC sampling
    best_param, all_params, log_likelihood = hmcmc_optimizer.optimize(
        config,
        num_results=num_results, burn_in_steps=num_burnin_steps
    )
    log.info(f"Best parameter: {best_param}, Log-likelihood: {log_likelihood}")
    hmcmc_optimizer.save_state('hmcmc_checkpoint')


if __name__ == '__main__':
    main()
