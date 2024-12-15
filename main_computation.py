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

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)
EXIT_CHECKPOINT = 'checkpoint_exit'


def parse_args():
    """
    -------------------------------------------------------
    Parse and validate command line arguments
    -------------------------------------------------------
    Returns:
       args - validated command line arguments (argparse.Namespace)
    -------------------------------------------------------
    """
    parser = argparse.ArgumentParser(
        description='Network Formation Model Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['mle', 'hmcmc', 'vi'],
        required=True,
        help='Estimation algorithm to use (mle, hmcmc, or vi)'
    )

    parser.add_argument(
        '--run_path',
        type=str,
        required=True,
        help='Path to store saved models and checkpoints'
    )

    parser.add_argument(
        '--load',
        action='store_true',
        help='Load previous state of HMCMC if available'
    )
    parser.add_argument(
        '--load_path',
        type=str,
        help='Path to load previous state of Network formation model'
    )
    parser.add_argument(
        '--state_type',
        type=str,
        choices=['hmcmc', 'vi'],
        help='Whether the run that saved the state was HMCMC or VI'
    )

    args = parser.parse_args()

    # Validate run_path
    run_path = args.run_path
    exit_checkpoint = os.path.join(run_path, EXIT_CHECKPOINT)
    if not os.path.isdir(run_path) or not os.path.isdir(exit_checkpoint):
        try:
            os.makedirs(run_path, exist_ok=True)
            os.makedirs(exit_checkpoint, exist_ok=True)
            log.info(f"Created directory {run_path}")
        except Exception as e:
            parser.error(f"Could not create directory {run_path}: {str(e)}")

    if args.load:
        load_path = args.load_path
        assert os.path.isdir(load_path), f"Invalid load path: {load_path}"
        assert args.state_type in ['hmcmc', 'vi'], f"Invalid state type: {args.state_type}"
        # Check if state file exists
        hmcmc_state = os.path.join(load_path, 'hmcmc_state.pkl')
        vi_state = os.path.join(load_path, 'vi_state.pkl')
        assert os.path.isfile(hmcmc_state) or os.path.isfile(vi_state), f"No state found in {load_path}"

    return args


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
    lr = 1e-3
    clip_val = 10

    if args.algorithm == 'mle' or args.algorithm == 'hmcmc':
        # Define MLE optimizer object
        hmcmc_optimizer = MyHMCMC(
            num_dims_theta=num_dims_theta,
            num_dims_h=num_dims_h,
            num_chains=1,
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
            num_epochs = 10
            # Optimize the likelihood function with MLE
            log.info("Optimizing likelihood function with MLE")
            losses = hmcmc_optimizer.optimize_w_mle(config, run_path=args.run_path, num_epochs=num_epochs)
            log.info(f"Log-likelihood: {losses[-1]}: MLE theta: {hmcmc_optimizer.param_state}")
            hmcmc_optimizer.save_state(args.run_path)
        else:
            num_results = 20
            num_burnin_steps = 100
            # Run HMC sampling
            best_param, all_params, log_likelihood = hmcmc_optimizer.optimize(
                config, num_results=num_results, burn_in_steps=num_burnin_steps
            )
            log.info(f"Best parameter: {best_param}, Log-likelihood: {log_likelihood}")
            hmcmc_optimizer.save_state(args.run_path)
    elif args.algorithm == 'vi':
        num_epochs = 20
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
        log.info(f"Loss/ELBO: {-losses[-1]}: VI theta: {params}")
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
