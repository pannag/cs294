#!/bin/bash

python train_bc.py expert-data-experts.RoboschoolWalker2d-v1.pkl --model RoboschoolWalker2d-mlp-dagger0 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger0
python data_combiner.py expert-data-experts.RoboschoolWalker2d-v1.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-1.RoboschoolWalker2d.pkl
python train_bc.py dagger-1.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger1 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger1
python data_combiner.py dagger-1.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-2.RoboschoolWalker2d.pkl
python train_bc.py dagger-2.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger2 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger2
python data_combiner.py dagger-2.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-3.RoboschoolWalker2d.pkl
python train_bc.py dagger-3.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger3 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger3
python data_combiner.py dagger-3.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-4.RoboschoolWalker2d.pkl
python train_bc.py dagger-4.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger4 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger4
python data_combiner.py dagger-4.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-5.RoboschoolWalker2d.pkl
python train_bc.py dagger-5.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger5 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger5
python data_combiner.py dagger-5.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-6.RoboschoolWalker2d.pkl
python train_bc.py dagger-6.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger6 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger6
python data_combiner.py dagger-6.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-7.RoboschoolWalker2d.pkl
python train_bc.py dagger-7.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger7 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger7
python data_combiner.py dagger-7.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-8.RoboschoolWalker2d.pkl
python train_bc.py dagger-8.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger8 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger8
python data_combiner.py dagger-8.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-9.RoboschoolWalker2d.pkl
python train_bc.py dagger-9.RoboschoolWalker2d.pkl --model RoboschoolWalker2d-mlp-dagger9 --train
python run_expert.py experts/RoboschoolWalker2d-v1.py --num_rollouts 20 --trained_policy RoboschoolWalker2d-mlp-dagger9
python data_combiner.py dagger-9.RoboschoolWalker2d.pkl dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-10.RoboschoolWalker2d.pkl
