#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=186G
#SBATCH --time=06:00:00 # set to 4h
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=arthurdehgan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -J train_net_subclf400
#SBATCH --output="/home/kikuko/slurm_logs/LOG-%x-%j.out"

module --quiet load python/3.11.5

source /home/kikuko/.meegnet/bin/activate

python /home/kikuko/meegnet/scripts/train_net.py --config /home/kikuko/meegnet/scripts/belugasub.ini --max-subj=400

