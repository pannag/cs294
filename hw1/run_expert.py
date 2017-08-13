#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib
import action_pred_module as ap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--trained_policy', type=str, default=None, 
                        help='Trained policy to use instead of expert policy.')
    args = parser.parse_args()

    print('loading expert policy')
    module_name = args.expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded', module_name)

    env, policy = policy_module.get_env_and_policy()
    max_steps = args.max_timesteps or env.spec.timestep_limit
    # get our trained policy predictor
    action_pred = None
    if args.trained_policy is not None:
        action_pred = ap.ActionPredictor(model_name=args.trained_policy)
        action_pred.start()

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            if args.trained_policy is not None:
                # obs is shape (I, ). Need to convert to [1, I] before passing into TF
                action = action_pred.predict_action(input=obs.reshape(-1, obs.shape[0]))
                action = action.reshape(-1)  # reshape to 1-D array
            else:
                action = policy.act(obs)

            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    if action_pred is not None:
        action_pred.stop()

    if args.trained_policy is None:
        expert_data = {'observations': np.array(observations),
                        'actions': np.array(actions)}
        outfile = 'expert-data-' + module_name + '.pkl'
        print('Ran the expert policy. Saving the training data in ', outfile)
        with open(outfile, 'wb') as f:
            pickle.dump(expert_data, f)

if __name__ == '__main__':
    main()
