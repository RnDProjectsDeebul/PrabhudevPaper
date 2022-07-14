#!/bin/bash
#SBATCH --job-name=gdpa
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=120GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /scratch/pbenga2s/Pytorch/Evidential_uncertainty/pytorch-classification-uncertainty/Output/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /scratch/pbenga2s/Pytorch/Evidential_uncertainty/pytorch-classification-uncertainty/Output/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pytorch_projects

# locate to your root directory
cd /scratch/pbenga2s/Pytorch/Evidential_uncertainty/pytorch-classification-uncertainty
# run the script
python main.py --train --dropout --uncertainty --mse --epochs 50

# python main.py --train --dropout --uncertainty --mse --epochs 100
# python main.py --test --uncertainty --mse
# python main.py --test --uncertainty --mse



#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml