# Network Formation Model Estimation

This project implements estimation methods for strategic network formation models described in Menzel (2016). The focus is on modeling strategic network formation with many agents, providing three estimation methods:

- Maximum Likelihood Estimation (MLE)
- Hamiltonian Monte Carlo (HMCMC) 
- Variational Inference (VI)

The implementation supports various network statistics, node attributes, and handles both endogenous and exogenous interaction effects. This code is particularly tailored for analyzing the NYSE sponsor network, examining trust relationships and strategic link formation patterns between market participants.

## Setting Up the Environment (Singularity)
Note: The following instructions are for running the code on a NYU HPC cluster using Singularity containers. Skip this section if you are running the code locally.

### Singularity Containers
Pre-requisite is to have created Singularity containers and overlay files located at in the scratch folder, for instance:
```
/scratch/$USER/environments/singularity1/
/scratch/$USER/environments/singularity2/
/scratch/$USER/environments/singularity3/
```

Make sure all necessary overlay files are present in these directories before running experiments. 
See more details on singularity [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)

### 1. Request Compute Resources
```bash
srun -c8 --mem=96000 -t2:00:00 --pty /bin/bash
```

### 2. Start Singularity Container
Note: 
- Replace the Singularity container path with the appropriate container path. 
- Make sure to use the `--nv` flag for GPU support.
- make sure the overlay file ends with `:rw` instead of `:ro` for read-write access (if we have not installed the necessary packages in the container).
```bash
singularity exec --nv \
    --overlay /scratch/$USER/environment/singularity1/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash
```

### 3. Configure Environment
```bash
# Source environment files
source /ext/env.sh  # or source /ext3/env.sh
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:oyew707/project_broadway.git
cd project_broadway
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- TensorFlow (2.x)
- TensorFlow Probability 
- NumPy
- Pandas

## Data Requirements

The code expects data files to be located in a `./data` directory with the following files:

### Required Files
- `nyse_node_sp1.csv`: Node attributes data
- `nyse_edge_buy_sp_sp1.csv`: Edge/transaction data
- `nyse_edge_buy_com1.csv`: Committee membership data

### File Structure

#### Node Data (nyse_node_sp1.csv)
```
Columns:
- name: Name of the member
- ever_committee: Whether member ever served on committee (0/1)
- node_id: Unique identifier for the member
- ethnicity: Ethnic group classification
- ever_sponsor: Whether member ever served as sponsor (0/1)
```

#### Edge Data (nyse_edge_buy_sp_sp1.csv)
```
Columns:
- buyer_id: ID of seat buyer
- sponsor1_id: ID of first sponsor
- sponsor2_id: ID of second sponsor
- f1, f2, f3, f4: Transaction features
- blackballs: Number of negative votes
- whiteballs: Number of positive votes
- year: Year of transaction
```

#### Committee Data (nyse_edge_buy_com1.csv)
```
Columns:
- buyer_id: ID of seat buyer
- committee_id: ID of committee member
- f1, f2, f3, f4: Transaction features
- blackballs: Number of negative votes
- whiteballs: Number of positive votes
- year: Year of transaction
```

## Command Line Parameters

The code can be run using `main_computation.py` with the following arguments:

```bash
python main_computation.py \
    --algorithm [mle|hmcmc|vi] \
    --run_path PATH \
    [optional arguments]
```

### Required Arguments:
- `--algorithm`: Estimation method to use (`mle`, `hmcmc`, or `vi`)
- `--run_path`: Directory to save models and checkpoints

### Optional Arguments:
- `--load`: Load previous model state
- `--load_path`: Path to load previous model state
- `--state_type`: Type of state to load (`hmcmc` or `vi`)
- `-lr, --learning_rate`: Learning rate (default: 1e-4)
- `-cv, --clip_value`: Gradient clipping value (default: 5.0)
- `-me, --maximum_epochs`: Maximum training epochs (default: 10)
- `-nc, --num_chains`: Number of HMCMC chains (default: 5)
- `-nr, --num_results`: Number of samples to draw (default: 5)
- `--burn_in`: HMCMC burn-in steps (default: 5)
- `--leap_frog`: HMCMC leap frog steps (default: 2)

Note: total_evaluations_per_chain = (num_results//num_chains + burn_in) * leap_frog * 2

### Environment Variables:
- LOGLEVEL: Set logging level (default: INFO)
- RANDOMSEED: Set random seed for reproducibility 

Example usage:
```bash
LOGLEVEL=debug RANDOMSEED=1234 python main_computation.py \
    --algorithm hmcmc \
    --run_path ./results \
    --num_chains 10 \
    --num_results 1000 \
    --burn_in 100
```

## References

[1] Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent Space Approaches to Social Network Analysis. Journal of the American Statistical Association, 97(460), 1090-1098.

[2] Menzel, K. (2016). Strategic Network Formation with Many Agents. Working Paper.

[3] Robert, C. P., & Casella, G. (2004). The Metropolis-Hastings Algorithm. In Monte Carlo Statistical Methods (pp. 267-318). Springer.

[4] Brooks, S. (2011). MCMC Using Hamiltonian Dynamics. In Handbook of Markov Chain Monte Carlo (Chapter 5). CRC Press.