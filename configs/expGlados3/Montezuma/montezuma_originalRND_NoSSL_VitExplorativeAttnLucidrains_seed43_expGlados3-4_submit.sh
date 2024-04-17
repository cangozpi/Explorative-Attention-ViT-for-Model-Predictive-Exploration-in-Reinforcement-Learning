#!/bin/bash 
# --job-name=Montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed43_expGlados3-4_7day_batchjob

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################
# Load Anaconda
echo "======================="
echo "Activating conda env: att"
conda activate att
echo ""
echo "======================================================================================"


echo
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo
echo

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script command below
#python3 main.py --train --config_path=./configs/CartPole/config_rnd00.conf --log_name=CartPole_rnd00_5hr --save_model_path=checkpoints/CartPole/rnd00_5hr.ckpt

cd ../../../

torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=34 main.py --train --num_env_per_process 64 --config_path=./configs/expGlados3/Montezuma/config_originalRND_NoSSL_VitExplorativeAttnLucidrains.conf --log_name=montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed43_expGlados3-4 --save_model_path=checkpoints/expGlados3/Montezuma/montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed43_expGlados3-4.ckpt --seed=43 --gpu_id=3 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8