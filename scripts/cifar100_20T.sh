python main.py --model dae --dataset seq-cifar100 --mode buf_kd --device cuda --total_tasks 20 --num_tasks 20 --norm_type bn_track_affine --lr 0.01 --lr_score 0.05 --lamb 1.05 --alpha 1 --beta 1 --buffer_size 2000 --dropout 0.2 --sparsity 0.8 --seed 0 --batch_size 32 --val_batch_size 256 --verbose --amp --resume --gpu_id 0