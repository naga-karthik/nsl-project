#python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer adam --exp_id exp1 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit  
python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 50 --optimizer adamw --exp_id cif10_6_pre --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit 
python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 50 --optimizer adamw --exp_id cif10_6_post --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit --block postnorm
#python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer sgd --exp_id exp3 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
#python run_exp_vit.py --model vit --layers 2 --batch_size 128 --epochs 1 --optimizer momentum --exp_id exp4 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit

#python run_exp_vit.py --model vit --layers 4 --batch_size 128 --epochs 1 --optimizer adamw --exp_id exp5 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
#python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 1 --optimizer adamw --exp_id exp6 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
#python run_exp_vit.py --model vit --layers 6 --batch_size 128 --epochs 1 --optimizer adamw --block postnorm --exp_id exp7 --log --log_dir /home/mila/v/venkatesh.ramesh/scratch/as2/IFT6135-2/assignment2/logs_vit
