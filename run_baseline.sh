#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=40
#SBATCH --mem=700G
#SBATCH --time=48:00:00 

module --quiet load python/3.7

source /home/kikuko/camcan/bin/activate

python /home/kikuko/scripts/camcan/baseline_ml.py --mode=overwrite --path=/home/kikuko/projects/def-kjerbi/kikuko/data/ --save=/home/kikuko/scripts/results/ --model-name=baseline_$1 -e="ALL" --log --feature=$2 --classifier=$1 --space='riemannian' --hypop=$3 --max-subj=200

