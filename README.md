# Robust ensembles for valid inference with invalid instruments

Code to reproduce the experiments.

`setup_env.sh` will install create a conda environment called `robust` and install all the required packages.

The experiments can be run by running `demand.py` and `mendel.py` with the appropriate hyperparameters.

Both scripts are designed to be run on a slurm cluster to search over a large number of experimental settings (random seed, bias parameters, number of valid instruments, etc.). For convienience, `gen_experiments.py` can create the slurm job files. They can, however be run on an individual machine using the call strings below.

## Demand experiments

Each of the demand experiments can be run using variants of the following call strings:

`python demand.py --seed 1 -n 100000 --n_inst 8 --n_valid 6 --bias 1 --epochs 1`

Figure 1 in the paper sweeps the bias parameter from 1 to 4 and the number of valid instruments from 5 to 8 across 50 random seeds. To generate a slurm file to run this experiment, you can run

`python gen_experiments.py 'python demand.py --seed {1-50} -n 100000 --n_inst 8 --n_valid {5,6,7,8} --bias {1-4} --epochs 1'`

This will be 800 jobs, each of which will train 10 deep networks, so you need a cluster to run these (or a lot of patience!). Running the above command with `--local` flag will produce a shell script to run the experiments sequentially on a single machine.

Figure 2 and 3 can be reproduced by working with the saved output from these experiments.

## Mendelian Randomization experiments

Each of the demand Mendelian Randomization can be run using variants of the following call strings:

`python mendel.py -n 400000 --n_inst 100 --n_valid 60 --seed 21 --linear_response --var_scale 1. --batch_size 1000 --model_id 0`

This script optionally takes a `model_id` which trains each member of the ensemble separately (the flag refers to which ensemble member is being trained; running without this flag will train the full ensemble sequentially) so that they can be run in parallel. Running this script in sequential mode requires training 102 deep networks (100 ensemble members + the two baselines), so it's strongly advised to run this on a cluster. As before, the cluster job files can be generated with the `gen_experiments.py` script.

`python gen_experiments.py 'python mendel.py -n 400000 --n_inst 100 --n_valid {50,60,70,80,90,100} --seed {1-30} --linear_response --var_scale 1. --batch_size 1000 --uid SLURM_ARRAY_TASK_ID --model_id {0-99}'`

The above command will generate 18000 jobs, but can be used to reproduce the results in table 1 and 2 (of course, running with an individual seed and set of parameters will require far less compute and should be sufficient to confirm the reported results are accurate). 

