#!/bin/sh

#SBATCH --job-name=WingsNet  # Job name
#SBATCH --output=/home/guests/jorge_padilla/output/logs/WingsNet-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/home/guests/jorge_padilla/output/logs/WingsNet-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=1  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)

# load python module
source /home/guests/jorge_padilla/miniconda3/etc/profile.d/conda.sh

# Set your wandb API key
# export WANDB_API_KEY=$(cat ./wandb_api_key.txt)

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate WingsNet

python -u ./main.py

conda deactivate