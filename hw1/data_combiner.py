#!/usr/bin/env python

"""
Code to load multiple expert policy generated data and combine.
Example usage:
python data_combiner.py expert-data-experts.RoboschoolWalker2d-v1.pkl\
     dagger-experts.RoboschoolWalker2d-v1.pkl --output dagger-1.RoboschoolWalker2d.pkl

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
        print(args.data_filename1, "actions = {}, obs = {}".format(actions.shape, obs.shape))

    with open(args.data_filename2, 'rb') as f:
        expert_data = pickle.load(f)
        obs = expert_data.get('observations')
        actions = expert_data.get('actions')
        all_obs.append(obs)
        all_actions.append(actions)
        print(args.data_filename2, "actions = {}, obs = {}".format(actions.shape, obs.shape))
    
    with open(args.output, 'wb') as f:
        print('writing combined expert data')
        out_obs = np.concatenate((all_obs[0], all_obs[1]), axis=0)
        out_actions = np.concatenate((all_actions[0], all_actions[1]), axis=0)
        expert_data = {'observations': out_obs, 'actions': out_actions}
        print(args.output, "actions = {}, obs = {}".format(out_actions.shape, out_obs.shape))
        pickle.dump(expert_data, f)


if __name__ == '__main__':
    main()



