#!/usr/bin/env python

"""
Code to load multiple expert policy generated data and combine.
Example usage:
    python train_bc.py expert-data-experts.RoboschoolWalker2d-v1.pkl \
        --model bc.RoboschoolWalker2d-v1-linear --train
    python train_bc.py expert-data-experts.RoboschoolHalfCheetah-v1.pkl \
        --model bc.RoboschoolHalfCheetah-v1-3layer --model_type_mlp --train
"""


import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename1', type=str)
    parser.add_argument('data_filename2', type=str)
    parser.add_argument('--output', type=str, default='data')

    args = parser.parse_args()

    print('loading expert data')

    all_obs = []
    all_actions = []
    with open(args.data_filename1, 'rb') as f:
        expert_data = pickle.load(f)
        obs = expert_data.get('observations')
        actions = expert_data.get('actions')
        all_obs.append(obs)
        all_actions.append(actions)

        obs_size = obs.shape[1]
        actions_size = actions.shape[1]
        num_samples = obs.shape[0]
        print(args.data_filename1, ": num_samples = {}, actions_size = {}, obs_size = {}".format(num_samples, actions_size, obs_size))

    with open(args.data_filename2, 'rb') as f:
        expert_data = pickle.load(f)
        obs = expert_data.get('observations')
        actions = expert_data.get('actions')
        all_obs.append(obs)
        all_actions.append(actions)

        obs_size = obs.shape[1]
        actions_size = actions.shape[1]
        num_samples = obs.shape[0]
        print(args.data_filename2, ": num_samples = {}, actions_size = {}, obs_size = {}".format(num_samples, actions_size, obs_size))
    
    with open(args.output, 'wb') as f:
        print('writing combined expert data')
        out_obs = np.concatenate((all_obs[0], all_obs[1]), axis=0)
        out_actions = np.concatenate((all_actions[0], all_actions[1]), axis=0)
        expert_data = {'observations': out_obs, 'actions': out_actions}
        obs_size = out_obs.shape[1]
        actions_size = out_actions.shape[1]
        num_samples = out_obs.shape[0]
        print(args.data_filename2, ": num_samples = {}, actions_size = {}, obs_size = {}".format(num_samples, actions_size, obs_size))

        pickle.dump(expert_data, f)


if __name__ == '__main__':
    main()



