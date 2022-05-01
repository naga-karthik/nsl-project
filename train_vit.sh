#!/bin/bash

# python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer adam --exp_id exp1 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit  
# python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 50 --optimizer adamw --exp_id cif10_6_pre --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit 
# python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 50 --optimizer adamw --exp_id cif10_6_post --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit --block postnorm
# python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer sgd --exp_id exp3 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
# python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer momentum --exp_id exp4 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit

# CIFAR 100 experiments
results_path='/home/GRAMES.POLYMTL.CA/u114716/neural_scaling_laws/nsl-project/results' 
echo "###### PRE-NORM EXPERIMENT STARTED ########"
CUDA_VISIBLE_DEVICES=0,1 python run_exp_vit.py --dataset places365 --model vit --layers 6 --block prenorm -bs 256 -ps 4 --epochs 200 --opt adam --print_every 1000 --exp_id pla365_6_pre_bs=128_ps=4_ep=200_adam --log --log_dir $results_path 
echo "###### PRE-NORM EXPERIMENT ENDED ########"
echo " "
# echo "###### POST-NORM EXPERIMENT STARTED ########"
# python run_exp_vit.py --dataset cifar100 --model vit --layers 2 --block postnorm -bs 256 -ps 4 --epochs 200 --opt adam --exp_id cif100_2_post_bs=256_ps=4_ep=200_adam --log --log_dir $results_path 
# echo "###### POST-NORM EXPERIMENT ENDED ########"
# #python run_exp_vit.py --model vit --layers 4 --batch_size 128 --epochs 1 --optimizer adamw --exp_id exp5 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
# #python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 1 --optimizer adamw --block postnorm --exp_id exp7 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
