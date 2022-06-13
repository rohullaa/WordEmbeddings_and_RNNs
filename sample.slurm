#!/bin/bash

#SBATCH --job-name=test_rnn
#SBATCH --account=nn9851k
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks-per-node=3

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-nlptools/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-pytorch/1.7.1-gomkl-2019b-cuda-10.1.243-Python-3.7.4
module load nlpl-gensim/3.8.3-gomkl-2019b-Python-3.7.4

python3 evaluate.py --model RNNs --rnn_type LSTM --which_state mean --input_size 25 --which_state max --bidirectional True
