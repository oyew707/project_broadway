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
import argparse
import os
from src.dataset import load_data, process_data
from src.likelihood import calculate_weights, LikelihoodConfig
from src.HMCMC import MyHMCMC
from src.logger import getlogger
import atexit
from src.reparameterizationVI import ReparameterizationVI
from src.parser import parse_args, EXIT_CHECKPOINT

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)


def run_estimation(args: argparse.Namespace, config: LikelihoodConfig):
    """
    -------------------------------------------------------
    Run the specified estimation algorithm
    -------------------------------------------------------
    Parameters:
        args - command line arguments (argparse.Namespace)
        config - likelihood configuration (LikelihoodConfig)
    -------------------------------------------------------
    """
    log.info(f"Running {args.algorithm} estimation algorithm")

    # Define common input parameters
    x_dim = len(list(config.node_attrs.values())[0])
    s_dim = 1  # len(list(node_stats.values())[0])
    num_dims_theta = 2 * (x_dim + s_dim)
    num_dims_h = 1
    lr = args.learning_rate
    clip_val = args.clip_value

    if args.algorithm == 'mle' or args.algorithm == 'hmcmc':
        # Define MLE optimizer object
        hmcmc_optimizer = MyHMCMC(
            num_dims_theta=num_dims_theta,
            num_dims_h=num_dims_h,
            num_chains=args.num_chains,
            verbosity=True,
            lr=lr,
            clip_val=clip_val
        )
        # Save state at exit in case of crash
        atexit.register(hmcmc_optimizer.save_state, os.path.join(args.run_path, EXIT_CHECKPOINT))

        if args.load:
            log.info("Loading previous state of HMCMC")
            hmcmc_optimizer.load_state(args.load_path, config, state_type=args.state_type)

        if args.algorithm == 'mle':
            num_epochs = args.maximum_epochs
            # Optimize the likelihood function with MLE
            log.info("Optimizing likelihood function with MLE")
            losses = hmcmc_optimizer.optimize_w_mle(config, run_path=args.run_path, num_epochs=num_epochs)
            log.info(f"Negative Log-likelihood: {losses[-1]}: MLE theta: {hmcmc_optimizer.param_state}")
            hmcmc_optimizer.save_state(args.run_path)
        else:
            num_results = args.num_results
            num_burnin_steps = args.burn_in
            # Run HMC sampling
            best_param, all_params, log_likelihood = hmcmc_optimizer.optimize(
                config, num_results=num_results, burn_in_steps=num_burnin_steps
            )
            log.info(f"Best parameter: {best_param}, Log-likelihood: {log_likelihood}")
            hmcmc_optimizer.save_state(args.run_path)
    elif args.algorithm == 'vi':
        num_epochs = args.maximum_epochs
        # Define VI optimizer object
        vi_optimizer = ReparameterizationVI(
            num_dims_theta=num_dims_theta,
            num_dims_h=num_dims_h,
            verbose=True,
            lr=lr,
            clip_val=clip_val,
            num_epochs=num_epochs,
            num_samples=1
        )
        # Save state at exit in case of crash
        atexit.register(vi_optimizer.save_state, os.path.join(args.run_path, EXIT_CHECKPOINT))

        if args.load:
            log.info("Loading previous state of VI")
            vi_optimizer.load_state(args.load_path, config, state_type=args.state_type)

        # Optimize the likelihood function with VI
        log.info("Optimizing likelihood function with VI")
        losses = vi_optimizer.optimize(config, run_path=args.run_path, optimize_alpha=True)
        params = vi_optimizer.variational_params['mean']
        log.info(f"ELBO: {-losses[-1]}: VI theta: {params}")
        vi_optimizer.save_state(args.run_path)
    else:
        log.error(f"Invalid algorithm: {args.algorithm}")
        raise ValueError(f"Invalid algorithm: {args.algorithm}")



def main():

    # Parse command line arguments
    args = parse_args()

    # Load and Preprocess data
    node_df, edge_df, committee_df = load_data(data_path='./data')
    node_attrs, transactions, network_stats, edges = process_data(node_df, edge_df, committee_df)
    node_stats = {k: v['degree'] for k, v in network_stats.items()}

    # Calculate importance weights
    weights = {node_id: calculate_weights(s > 0, node_stats.values()) for node_id, s in node_stats.items()}

    # Likelihood Config
    config = LikelihoodConfig(node_attrs=node_attrs, node_stats=node_stats, edges=edges, weights=weights)

    try:
        run_estimation(args, config)
    except Exception as e:
        log.error(f"Error running estimation: {str(e)}")
        raise e


if __name__ == '__main__':
    main()
