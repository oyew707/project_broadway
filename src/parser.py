"""
-------------------------------------------------------
File for parsing command line arguments.
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "12/15/24"
-------------------------------------------------------
"""

# Imports
import argparse
import os
from src.logger import getlogger

# Constants
loglevel = os.getenv('LOGLEVEL', 'INFO').lower()
log = getlogger(__name__, loglevel)
EXIT_CHECKPOINT = 'checkpoint_exit'


def validate_args(args: argparse.Namespace):
    """
    -------------------------------------------------------
    Validate command line arguments
    -------------------------------------------------------
    Parameters:
       args - command line arguments (argparse.Namespace)
    -------------------------------------------------------
    """
    # Validate algorithm
    alg = args.algorithm
    assert alg in ['mle', 'hmcmc', 'vi'], f"Invalid algorithm: {args.algorithm}"

    # Validate run_path
    run_path = args.run_path
    exit_checkpoint = os.path.join(run_path, EXIT_CHECKPOINT)
    if not os.path.isdir(run_path) or not os.path.isdir(exit_checkpoint):
        try:
            os.makedirs(run_path, exist_ok=True)
            os.makedirs(exit_checkpoint, exist_ok=True)
            log.info(f"Created directory {run_path}")
        except Exception as e:
            log.error(f"Could not create directory {run_path}: {str(e)}")
            raise e

    if args.load:
        load_path = args.load_path
        assert os.path.isdir(load_path), f"Invalid load path: {load_path}"
        assert args.state_type in ['hmcmc', 'vi'], f"Invalid state type: {args.state_type}"
        # Check if state file exists
        hmcmc_state = os.path.join(load_path, 'hmcmc_state.pkl')
        vi_state = os.path.join(load_path, 'vi_state.pkl')
        assert os.path.isfile(hmcmc_state) or os.path.isfile(vi_state), f"No state found in {load_path}"

    # Validate learning rate
    assert args.learning_rate > 0, f"Invalid learning rate: {args.learning_rate}"

    # Validate clip value
    assert args.clip_value > 0, f"Invalid clip value: {args.clip_value}"

    # Validate maximum epochs
    if alg in ['mle', 'vi']:
        assert args.maximum_epochs > 0, f"Invalid maximum epochs: {args.maximum_epochs}"

    if alg == 'hmcmc':
        # Validate number of chains
        assert args.num_chains > 0, f"Invalid number of chains: {args.num_chains}"

        # Validate number of results
        assert args.num_results > 0, f"Invalid number of results: {args.num_results}"
        # num_results//num_chains samples are drawn from each chain

        # Validate burn-in
        assert args.burn_in >= 0, f"Invalid burn-in: {args.burn_in}"

        # Validate leap frog
        assert args.leap_frog >= 0, f"Invalid leap frog: {args.leap_frog}"
        # Leap frog of 0 will mean sample in the same chain are correlated


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
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for optimization'
    )
    parser.add_argument(
        '-cv', '--clip_value',
        type=float,
        default=5.0,
        help='Absolute value of max and min gradient values'
    )
    parser.add_argument(
        '-me', '--maximum_epochs',
        type=int,
        default=10,
        help='Maximum number of epochs for optimization'
    )
    parser.add_argument(
        '-nc', '--num_chains',
        type=int,
        default=5,
        help='Number of chains for HMCMC'
    )
    parser.add_argument(
        '-nr', '--num_results',
        type=int,
        default=5,
        help='Number of samples to draw from HMCMC'
    )
    parser.add_argument(
        '-bi', '--burn_in',
        type=int,
        default=5,
        help='Burn-in steps for HMCMC before drawing samples'
    )
    parser.add_argument(
        '-lpf', '--leap_frog',
        type=int,
        default=2,
        help='Leap frog steps for HMCMC; how many steps to skip before drawing next sample'
    )

    args = parser.parse_args()
    # Validate arguments
    validate_args(args)

    return args
