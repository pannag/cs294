#!/usr/bin/env python

"""
Code to run dagger over multiple iterations.

Note: right now this only is useful to generate the commands, and the 
commands have to be run manually in a terminal on1 by one. 
If you try to run it all from this script, it gives an error.

Example usage:
    python run_dagger.py --module RoboschoolWalker2d --model mlp --num_iterations 10
    python run_dagger.py --module RoboschoolHopper --model mlp --num_rollouts 2

"""

import argparse
import os

NUM_DAGGER_ITERATIONS = 8
NUM_ROLLOUTS=2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help="One of the envs supported w/o the -v1.py suffix.")
    parser.add_argument('--model', type=str,
                        help="Only used for saving output. This dagger models will be saved as module-dagger-model*")
    parser.add_argument('--num_iterations', type=int, default=NUM_DAGGER_ITERATIONS, help='number of dagger iterations')
    parser.add_argument('--num_rollouts', type=int, default=NUM_ROLLOUTS, help='number of rollouts of the env.')
    parser.add_argument('--reuse_ckpt', action='store_true', default=False,
                        help='whether to use the model from previous dagger iteration as starting ckpt.')

    args = parser.parse_args()

    data_file = 'expert-data-experts.' + args.module + '-v1.pkl'
    model_name = args.module + '-' + args.model

    # First run the expert a couple of rollouts to get initial data.
    initial_expert_cmd="python run_expert.py experts/{}-v1.py --num_rollouts 2".format(args.module)
    print(initial_expert_cmd)
    init_ckpt=None
    for i in range(args.num_iterations):
        if init_ckpt is not None and args.reuse_ckpt:
            train_cmd = "python train_bc.py {} --model {}-dagger{} --train --init_ckpt={}".format(data_file, model_name, i, init_ckpt)
        else:
            train_cmd = "python train_bc.py {} --model {}-dagger{} --train".format(data_file, model_name, i)
        print(train_cmd)
        #os.system(train_cmd)

        run_expert_cmd = "python run_expert.py experts/{}-v1.py --num_rollouts {} --trained_policy {}-dagger{}".format(
            args.module, args.num_rollouts, model_name, i)
        print(run_expert_cmd)
        #os.system(run_expert_cmd)


        data_combiner_cmd = "python data_combiner.py {} dagger-experts.{}-v1.pkl --output dagger-{}.{}.pkl".format(data_file, args.module, i+1, args.module)
        print(data_combiner_cmd)
        #os.system(data_combiner_cmd)

        data_file = "dagger-{}.{}.pkl".format(i+1, args.module)
        #print("Combined new data_file: ", data_file)
        init_ckpt = "{}-dagger{}".format(model_name, i)
        #print("New init_ckpt: ", init_ckpt)



if __name__ == '__main__':
    main()

