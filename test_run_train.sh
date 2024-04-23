#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=arthurdehgan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -J train_net
#SBATCH --output="/home/kikuko/slurm_logs/LOG_test-%x-%j.out"

module --quiet load python/3.11.5

source /home/kikuko/.meegnet/bin/activate

python /home/kikuko/meegnet/scripts/train_net.py --config /home/kikuko/meegnet/scripts/beluga.ini 

