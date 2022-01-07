#!/bin/bash

export TOXIC_DIR=/toxic/language/data
export TASK_NAME=Toxic

export DATA=/media/lvzhiheng02/Windows-SSD/Users/cogito/data/civilComment
export RAN=42
export MODEL_DIR=./output_dir

python3 run_toxic.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased\
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --seed $RAN \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --data_dir $DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir $MODEL_DIR
