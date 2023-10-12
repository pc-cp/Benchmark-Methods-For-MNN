#!/bin/bash
# This file includes
# the hyperparameter settings for training and evaluating the individual algorithms.
# Note that you need to create two new folders in advance in the same directory as this file
# to store the file logs during pretraining and evaluation:
# -------> pretrain_output and linear_eval_output <-------
echo "pretrain moco(cifar10_1)"
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
wait
echo "evaluating moco(cifar10_1)"
#nohup python main_linear_eval.py \
#--name moco \
#--dataset cifar10 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar10_1 \
#>linear_eval_output/cifar10_1_1 2>&1 &
wait
echo "pretrain moco(cifar100_1)"
#nohup python main_pretrain.py \
#--name moco \
#--symmetric \
#--tem 0.2 \
#--momentum 0.99 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar100_1 \
#>pretrain_output/cifar100_1 2>&1 &
wait
echo "evaluating moco(cifar100_1)"
#nohup python main_linear_eval.py \
#--name moco \
#--dataset cifar100 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar100_1 \
#>linear_eval_output/cifar100_1_1 2>&1 &
wait
echo "pretrain moco(tin_1)"
#nohup python main_pretrain.py \
#--name moco \
#--symmetric \
#--tem 0.2 \
#--momentum 0.996 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--queue_size 16384 \
#--gpuid 0 \
#--seed 1339 \
#--logdir tin_1 \
#>pretrain_output/tin_1 2>&1 &
wait
echo "evaluating moco(tin_1)"
#nohup python main_linear_eval.py \
#--name moco \
#--dataset tinyimagenet \
#--gpuid 0 \
#--seed 1339 \
#--logdir tin_1 \
#>linear_eval_output/tin_1_1 2>&1 &
wait

echo "pretrain simclr(cifar10_2)"
#nohup python main_pretrain.py \
#--name simclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar10_2 \
#>pretrain_output/cifar10_2 2>&1 &
wait
echo "evaluating simclr(cifar10_2)"
#nohup python main_linear_eval.py \
#--name simclr \
#--dataset cifar10 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar10_2 \
#>linear_eval_output/cifar10_2_1 2>&1 &
wait
echo "pretrain simclr(cifar100_2)"
#nohup python main_pretrain.py \
#--name simclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar100_2 \
#>pretrain_output/cifar100_2 2>&1 &
wait
echo "evaluating simclr(cifar100_2)"
#nohup python main_linear_eval.py \
#--name simclr \
#--dataset cifar100 \
#--gpuid 0 \
#--seed 1339 \
#--logdir cifar100_2 \
#>linear_eval_output/cifar100_2_1 2>&1 &
wait
echo "pretrain simclr(tin_2)"
#nohup python main_pretrain.py \
#--name simclr \
#--symmetric \
#--tem 0.2 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--gpuid 0 \
#--seed 1339 \
#--logdir tin_2 \
#>pretrain_output/tin_2 2>&1 &
wait
echo "evaluating simclr(tin_2)"
#nohup python main_linear_eval.py \
#--name simclr \
#--dataset tinyimagenet \
#--gpuid 0 \
#--seed 1339 \
#--logdir tin_2 \
#>linear_eval_output/tin_2_1 2>&1 &
wait
echo "pretrain byol(cifar10_3)"
#nohup python main_pretrain.py \
#--name byol \
#--symmetric \
#--momentum 0.99 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_3 \
#>pretrain_output/cifar10_3 2>&1 &
wait
echo "evaluating byol(cifar10_3)"
#nohup python main_linear_eval.py \
#--name byol \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_3 \
#>linear_eval_output/cifar10_3_1 2>&1 &
wait
echo "pretrain byol(cifar100_3)"
#nohup python main_pretrain.py \
#--name byol \
#--symmetric \
#--momentum 0.99 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_3 \
#>pretrain_output/cifar100_3 2>&1 &
wait
echo "evaluating byol(cifar100_3)"
#nohup python main_linear_eval.py \
#--name byol \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_3 \
#>linear_eval_output/cifar100_3_1 2>&1 &
wait
echo "pretrain byol(tin_3)"
#nohup python main_pretrain.py \
#--name byol \
#--symmetric \
#--momentum 0.996 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_3 \
#>pretrain_output/tin_3 2>&1 &
wait
echo "evaluating byol(tin_3)"
#nohup python main_linear_eval.py \
#--name byol \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_3 \
#>linear_eval_output/tin_3_1 2>&1 &
wait
