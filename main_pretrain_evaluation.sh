#!/bin/bash
#nohup python main_pretrain.py \
#--name moco \
#--symmetric \
#--tem 0.2 \
#--momentum 0.99 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar10_1 \
#>pretrain_output/cifar10_1 2>&1 &

nohup python main_linear_eval.py \
--name moco \
--dataset cifar10 \
--gpuid 1 \
--seed 1339 \
--logdir cifar10_1 \
>linear_eval_output/cifar10_1_1 2>&1 &