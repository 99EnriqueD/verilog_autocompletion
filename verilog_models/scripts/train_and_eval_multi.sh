#!/bin/bash -l
#PBS -l walltime=24:00:00
#PBS -l partition=gpu
#PBS -l nodes=1:ppn=36:gpus=4:skylake
#PBS -N multi
#PBS -A default_project

cd $PBS_O_WORKDIR # Directory where this script is located
cd ..

# export TRANSFORMERS_CACHE="..."
# export HF_HOME="..."

# Train datasets: 'processed_train_unlabeled.csv' or 'processed_train_unlabeled_parsable.csv' or 'train_labeled.csv' (snippets) or 'data/train_labeled_parsable.csv'
# 3613 steps ~= 1/4 epoch
python train.py \
--train_csv 'data/processed_train_unlabeled.csv' \
--val_csv 'data/processed_val_unlabeled.csv' \
--output_dir 'outputs/multi' \
--model_path "Salesforce/codegen-350M-multi" \
--num_train_epochs 1 \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
--save_strategy "epoch" --eval_accumulation_steps 1 --evaluation_strategy "steps" --eval_steps 3613 \
--logging_first_step False --log_level 'info' --logging_strategy "steps" --logging_steps 3613 \
--weight_decay 0.1 --warmup_steps=500 --lr_scheduler_type "cosine" --learning_rate 5e-4 \

python evaluate_unlabeled.py \
--test_csv data/test_unlabeled.csv \
--model_path outputs/multi \
--output_dir outputs/multi/unlabeled_evaluation

python evaluate_labeled.py \
--test_csv data/test_labeled_randomized_pairs.csv \
--model_path 'outputs/multi' \
--output_dir 'outputs/multi/labeled_evaluation'
