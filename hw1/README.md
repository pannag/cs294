# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, OpenAI Gym, Roboschool v1.1

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* RoboschoolAnt-v1.py
* RoboschoolHalfCheetah-v1.py
* RoboschoolHopper-v1.py
* RoboschoolHumanoid-v1.py
* RoboschoolReacher-v1.py
* RoboschoolWalker2d-v1.py

Example usage:

All commands to be executed from the `hw1` folder.

To run dagger, use the `run_dagger.py` and run the commands printed out in a terminal 1-by-1.

 $ `python run_dagger.py --module RoboschoolWalker2d --model mlp --num_iterations 10 --num_rollouts 5`

Unfortuntely, calling other scripts in a single python script gives an error in
OpenAI Gym at the moment. In due time, I will write a module that does all in one file.

If you want to use the model from previous daggger iteration as the initial ckpt for next one, use `--reuse_ckpt`:

 $ `python run_dagger.py --module RoboschoolHopper --model mlp --num_iterations 10 --num_rollouts 5 --reuse_ckpt`

 See `run_dagger.py` for more details.

 To just run behavior cloning, first run the `run_expert.py` without giving any trained_policy
 and then run `train_bc.py` with the data file saved.

 $ `python run_expert.py experts/RoboschoolHumanoid-v1.py --render --num_rollouts 20`

 $ `python train_bc.py expert-data-experts.RoboschoolWalker2d-v1.pkl --model RoboschoolWalker2d-mlp-dagger0 --train --plot`

 Check the documentation of `train_bc.py` on how you can use either a linear or a MLP model. 
 All the network hyperparameters are currently declared as constants inside `train_bc.py`
 

