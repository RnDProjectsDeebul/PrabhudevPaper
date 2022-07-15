#!/bin/bash
#SBATCH --job-name=gdpa
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=60GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /scratch/pbenga2s/Pytorch/gdpa/output/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /scratch/pbenga2s/Pytorch/gdpa/output/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pytorch_projects

# locate to your root directory
cd /scratch/pbenga2s/Pytorch/gdpa
# run the script
python gdpa.py --dataset vggface --data_path Data --vgg_model_path saving_normal_way_model_Face_data224.pt --epochs 50 --patch_size 23



# python gdpa.py --dataset vggface --data_path LISA --vgg_model_path saving_normal_way_model_Traffic_data224.pt --epochs 50 --patch_size 71
# python gdpa.py --dataset vggface --data_path Data --vgg_model_path saving_normal_way_model_Face_data224.pt --epochs 50 --patch_size 71
# python gdpa.py --dataset imagenette --data_path Imagenette_OOD --vgg_model_path vgg16_model.pt --patch_size 50
# saving_normal_way_model_Imagenette_data224
# patch_sizes 71, 32, 23, 50

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml