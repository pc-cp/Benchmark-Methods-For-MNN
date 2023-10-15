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

echo "pretrain ressl(cifar10_4)"
#nohup python main_pretrain.py \
#--name ressl \
#--tem 0.04 \
#--momentum 0.99 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--weak \
#--queue_size 4096 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_4 \
#>pretrain_output/cifar10_4 2>&1 &
wait
echo "evaluating ressl(cifar10_4)"
#nohup python main_linear_eval.py \
#--name ressl \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_4 \
#>linear_eval_output/cifar10_4_1 2>&1 &
wait
echo "pretrain ressl(cifar100_4)"
#nohup python main_pretrain.py \
#--name ressl \
#--tem 0.04 \
#--momentum 0.99 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--weak \
#--queue_size 4096 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_4 \
#>pretrain_output/cifar100_4 2>&1 &
wait
echo "evaluating ressl(cifar100_4)"
#nohup python main_linear_eval.py \
#--name ressl \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_4 \
#>linear_eval_output/cifar100_4_1 2>&1 &
wait
echo "pretrain ressl(tin_4)"
#nohup python main_pretrain.py \
#--name ressl \
#--tem 0.04 \
#--momentum 0.996 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--weak \
#--queue_size 16384 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_4 \
#>pretrain_output/tin_4 2>&1 &
wait
echo "evaluating ressl(tin_4)"
#nohup python main_linear_eval.py \
#--name ressl \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_4 \
#>linear_eval_output/tin_4_1 2>&1 &
wait

echo "pretrain nnclr(cifar10_5)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 1 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_5 \
#>pretrain_output/cifar10_5 2>&1 &
wait
echo "evaluating nnclr(cifar10_5)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_5 \
#>linear_eval_output/cifar10_5_1 2>&1 &
wait
echo "pretrain nnclr(cifar100_5)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 1 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_5 \
#>pretrain_output/cifar100_5 2>&1 &
wait
echo "evaluating nnclr(cifar100_5)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_5 \
#>linear_eval_output/cifar100_5_1 2>&1 &
wait
echo "pretrain nnclr(tin_5)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--queue_size 16384 \
#--topk 1 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_5 \
#>pretrain_output/tin_5 2>&1 &
wait
echo "evaluating nnclr(tin_5)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_5 \
#>linear_eval_output/tin_5_1 2>&1 &
wait

echo "pretrain nnclr(cifar10_5a)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_5a \
#>pretrain_output/cifar10_5a 2>&1 &
wait
echo "evaluating nnclr(cifar10_5a)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_5a \
#>linear_eval_output/cifar10_5a_1 2>&1 &
wait
echo "pretrain nnclr(cifar100_5a)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_5a \
#>pretrain_output/cifar100_5a 2>&1 &
wait
echo "evaluating nnclr(cifar100_5a)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_5a \
#>linear_eval_output/cifar100_5a_1 2>&1 &
wait
echo "pretrain nnclr(tin_5a)"
#nohup python main_pretrain.py \
#--name nnclr \
#--symmetric \
#--tem 0.2 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--queue_size 16384 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_5a \
#>pretrain_output/tin_5a 2>&1 &
wait
echo "evaluating nnclr(tin_5a)"
#nohup python main_linear_eval.py \
#--name nnclr \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_5a \
#>linear_eval_output/tin_5a_1 2>&1 &
wait

echo "pretrain msf(cifar10_6)"
#nohup python main_pretrain.py \
#--name msf \
#--symmetric \
#--momentum 0.99 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--weak \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_6 \
#>pretrain_output/cifar10_6 2>&1 &
wait
echo "evaluating msf(cifar10_6)"
#nohup python main_linear_eval.py \
#--name msf \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_6 \
#>linear_eval_output/cifar10_6_1 2>&1 &
wait
echo "pretrain msf(cifar100_6)"
#nohup python main_pretrain.py \
#--name msf \
#--symmetric \
#--momentum 0.99 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--weak \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_6 \
#>pretrain_output/cifar100_6 2>&1 &
wait
echo "evaluating msf(cifar100_6)"
#nohup python main_linear_eval.py \
#--name msf \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_6 \
#>linear_eval_output/cifar100_6_1 2>&1 &
wait
echo "pretrain msf(tin_6)"
#nohup python main_pretrain.py \
#--name msf \
#--symmetric \
#--momentum 0.996 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--weak \
#--queue_size 16384 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_6 \
#>pretrain_output/tin_6 2>&1 &
wait
echo "evaluating msf(tin_6)"
#nohup python main_linear_eval.py \
#--name msf \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_6 \
#>linear_eval_output/tin_6_1 2>&1 &
wait

echo "pretrain snclr(cifar10_7)"
#nohup python main_pretrain.py \
#--name snclr \
#--symmetric \
#--tem 0.2 \
#--momentum 0.99 \
#--dataset cifar10 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_7 \
#>pretrain_output/cifar10_7 2>&1 &
wait
echo "evaluating snclr(cifar10_7)"
#nohup python main_linear_eval.py \
#--name snclr \
#--dataset cifar10 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar10_7 \
#>linear_eval_output/cifar10_7_1 2>&1 &
wait
echo "pretrain snclr(cifar100_7)"
#nohup python main_pretrain.py \
#--name snclr \
#--symmetric \
#--tem 0.2 \
#--momentum 0.99 \
#--dataset cifar100 \
#--aug_numbers 2 \
#--queue_size 4096 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_7 \
#>pretrain_output/cifar100_7 2>&1 &
wait
echo "evaluating snclr(cifar100_7)"
#nohup python main_linear_eval.py \
#--name snclr \
#--dataset cifar100 \
#--gpuid 1 \
#--seed 1339 \
#--logdir cifar100_7 \
#>linear_eval_output/cifar100_7_1 2>&1 &
wait
echo "pretrain snclr(tin_7)"
#nohup python main_pretrain.py \
#--name snclr \
#--symmetric \
#--tem 0.2 \
#--momentum 0.996 \
#--dataset tinyimagenet \
#--aug_numbers 2 \
#--queue_size 16384 \
#--topk 5 \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_7 \
#>pretrain_output/tin_7 2>&1 &
wait
echo "evaluating snclr(tin_7)"
#nohup python main_linear_eval.py \
#--name snclr \
#--dataset tinyimagenet \
#--gpuid 1 \
#--seed 1339 \
#--logdir tin_7 \
#>linear_eval_output/tin_7_1 2>&1 &
wait