#!/usr/bin/env python

"""
Code to load an expert policy generate data and train a model via behavioral cloning / dagger.
Example usage:
    python train_bc.py expert-data-experts.RoboschoolWalker2d-v1.pkl \
        --model bc.RoboschoolWalker2d-v1-linear --train --plot
    python train_bc.py expert-data-experts.RoboschoolHalfCheetah-v1.pkl \
        --model bc.RoboschoolHalfCheetah-v1-3layer --model_type_mlp --train --plot

Hyperparams are declared as constants at the top of the file.
"""


import pickle
import tensorflow as tf
import numpy as np
import itertools
import argparse
import random
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.contrib.slim as slim


LEARNING_RATE = 0.005 # 0.001
# How many times to go over the whole data
NUM_EPOCHS = 200
BATCH_SIZE = 64
N_HIDDEN_1 = 512
N_HIDDEN_2 = 512
N_HIDDEN_3 = 16


# Create model
def multilayer_perceptron(x, weights, biases, output_size):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1, name='hidden1')
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2, name='hidden2')
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3, name='hidden3')
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'], name='pred_action')

    return out_layer


def weights_and_biases(input_size, output_size):
    # Store layers weight & bias
    # Very important to use truncated_normal with stddev ~0.01. Else it will blow up.
    weights = {
        'h1': tf.Variable(tf.truncated_normal([input_size, N_HIDDEN_1], seed=1, stddev=0.01), name='weights1'),
        'h2': tf.Variable(tf.truncated_normal([N_HIDDEN_1, N_HIDDEN_2], seed=2, stddev=0.01), name='weights2'),
        'h3': tf.Variable(tf.truncated_normal([N_HIDDEN_2, N_HIDDEN_3], seed=3, stddev=0.01), name='weights3'),
        'out': tf.Variable(tf.truncated_normal([N_HIDDEN_3, output_size], seed=4, stddev=0.01), name='weights_out')
    }
    biases = {
        'b1': tf.Variable(tf.zeros([N_HIDDEN_1,]), name='biases1'),
        'b2': tf.Variable(tf.zeros([N_HIDDEN_2,]), name='biases2'),
        'b3': tf.Variable(tf.zeros([N_HIDDEN_3,]), name='biases3'),
        'out': tf.Variable(tf.zeros([output_size,]), name='biases_out')
    }
    return weights, biases


def linear_model(x, input_size, output_size):
    weights = tf.Variable(tf.truncated_normal([input_size, output_size], seed=1), name='weights')
    biases = tf.Variable(tf.truncated_normal([output_size], seed=2), name='biases')

    out_layer = tf.add(tf.matmul(x, weights), biases, name='pred_action')
    return out_layer, weights, biases


def setup_graph(input_size, output_size, model_type_mlp=False):
    x = tf.placeholder(tf.float32, [None, input_size], name='obs')
    y = tf.placeholder(tf.float32, [None, output_size], name='action')

    # Construct a MLP
    if model_type_mlp:
        weights, biases = weights_and_biases(input_size, output_size)
        pred = multilayer_perceptron(x, weights, biases, output_size)
    else:
        pred, weights, biases = linear_model(x, input_size, output_size)
    # Mean squared error
    diff = pred-y
    print("Pred: ", pred.get_shape(), " , y: ", y.get_shape(), " , diff: ", diff.get_shape())
    # cost = tf.reduce_mean(tf.pow(diff, 2)) # + tf.add_n([tf.nn.l2_loss(w) for _,w in weights.items()]) * 0.01
    cost = tf.losses.mean_squared_error(pred, y)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    if model_type_mlp:
        return {'pred':pred, 'cost': cost, 'train_op':optimizer, 'input':x, 'output':y, 'weights':weights['h1'], 'biases':biases['b1']}
    else:
        return {'pred':pred, 'cost': cost, 'train_op':optimizer, 'input':x, 'output':y, 'weights':weights, 'biases':biases}
    

def gen_batches(input, output, batch_size=BATCH_SIZE):
    """Generates batches of data over the input and outputs. 

       Assumes that the input and output arrays are of same size.
    """
    assert len(input) == len(output)
    n_samples = len(input)
    p = np.random.permutation(n_samples)
    shuffled_ip = input[p]
    shuffled_op = output[p]
    for i in np.arange(0, n_samples, batch_size):
        last_index = min(i+batch_size, n_samples)
        # print(i, ' : ', last_index, " of ", n_samples)
        yield shuffled_ip[i:last_index], shuffled_op[i:last_index]
        


def train(sess, g, input, output, model_name, batch_size=BATCH_SIZE, init_ckpt=None, plot=True, verbose=False):
    saver = None
    if init_ckpt is not None:
        saver = tf.train.import_meta_graph(init_ckpt + '.meta')
        saver.restore(sess, init_ckpt)
        print("Model restored:", init_ckpt)

    costs = []
    preds = []
    val_preds = []
    val_costs = []
    n_samples = input.shape[0]

    train_size = round((n_samples * 9) / 10)
    print('Train size : ', train_size, " of ", n_samples)
    train_input = input[:train_size, :]
    train_output = output[:train_size, :]
    val_input = input[train_size:, :]
    val_output = output[train_size:, :]
    for i in range(NUM_EPOCHS):
        for input_batch, output_batch in gen_batches(train_input, train_output, batch_size):
            pred_, cost_, _, weights_ = sess.run(
                [g['pred'], g['cost'], g['train_op'], g['weights']], 
                feed_dict={g['input']:input_batch, g['output']:output_batch})
        val_pred_, val_cost_ = sess.run(
            [g['pred'], g['cost']], 
            feed_dict={g['input']:val_input, g['output']:val_output})

        costs.append(cost_)
        preds.append(pred_)
        val_preds.append(val_pred_)
        val_costs.append(val_cost_)
        if i % (NUM_EPOCHS/10) == 0:
            print("Epoch", i, "Train Cost", cost_, "Val Cost", val_cost_)
    print("pred_ shape", pred_.shape)
    if verbose:
        print('Train Input stats\n', pd.DataFrame(train_input).describe())
        print('Preds stats\n', pd.DataFrame(pred_).describe())
        print('Output stats\n', pd.DataFrame(train_output).describe())
        print('Val Preds stats\n', pd.DataFrame(val_pred_).describe())
        print('Val Output stats\n', pd.DataFrame(train_output).describe())
        print('Learnt Weights\n', pd.DataFrame(weights_).describe())
    # Save model to disk.
    # Add ops to save and restore all the variables.
    if saver is None:
        saver = tf.train.Saver()
    saved_path = saver.save(sess, model_name)
    #output_filename = model_name + '.meta' 
    #tf.train.export_meta_graph(filename=output_filename)
    print("Model saved:", model_name)

    # Plot all results
    if plot:
        plt.figure()
        plt.plot(list(range(NUM_EPOCHS)), costs)
        plt.plot(list(range(NUM_EPOCHS)), val_costs, 'r')
        plt.legend(['Training', 'Validation'])
        plt.title('Cost vs Iteration num')
        plt.show()
    print("training done!")


def predict(sess, g, input, model_name):
    #saver = tf.train.Saver()
    #output_filename = model_name + '_' + 'model.ckpt' 
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(model_name + '.meta')
    saver.restore(sess, model_name)
    print("Model restored:", model_name)
    pred_ = sess.run([g['pred']], feed_dict={g['input']:input})
    return pred_


def predict_action(input, model_name):
    # define the tensorflow graph

    with tf.Graph().as_default():
        #tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_name + '.meta')

            graph = tf.get_default_graph()
            input_op = graph.get_tensor_by_name("obs:0") 
            pred_op = graph.get_tensor_by_name("pred_action:0")
            #weights_op = graph.get_tensor_by_name('weights1:0')
            # Initializing the variables
            init = tf.global_variables_initializer()
        
            # Start the session
            sess.run(init)
            saver.restore(sess, model_name)
            #print("Model restored:", model_name)
            pred_ = sess.run(pred_op, feed_dict={input_op:input})
            #print('Restored Weights', weights_)

            return pred_
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename', type=str, help='data to use for training purpose.')
    parser.add_argument('--init_ckpt', type=str, default=None, help='Initial checkpoint model.')
    parser.add_argument('--model', type=str, help='Output file for trained model.')
    parser.add_argument('--train', action='store_true', help='If False, just runs prediction.')
    parser.add_argument('--model_type_mlp', action='store_true', default=True, 
                        help='If False, uses linear model, otherwise MLP.')
    parser.add_argument('--plot', action='store_true', default=False, 
                        help='If true, plots the training and validation error against number of epochs')
   
    args = parser.parse_args()

    print('loading expert data')

    with open(args.data_filename, 'rb') as f:
        expert_data = pickle.load(f)
        obs_raw = expert_data.get('observations')
        actions_raw = expert_data.get('actions')

        # shuffle the data
        index = list(range(len(obs_raw)))
        random.shuffle(index)
        obs = obs_raw[index]
        actions = actions_raw[index]

        obs_size = obs.shape[1]
        actions_size = actions.shape[1]
        num_samples = obs.shape[0]
        print("num_samples = {}, actions_size = {}, obs_size = {}".format(num_samples, actions_size, obs_size))

                
    # define the tensorflow graph and train
    if args.train:
        print('Training now..')
        with tf.Graph().as_default():
            g = setup_graph(input_size=obs_size, output_size=actions_size, model_type_mlp=args.model_type_mlp)
            # Initializing the variables
            init = tf.global_variables_initializer()

            # Start the session
            with tf.Session() as sess:
                sess.run(init)
                # start an epoch
                train(sess,g,input=obs, output=actions, model_name=args.model, init_ckpt=args.init_ckpt, plot=args.plot)

                # Test that saved model works as intended
                # pred_ = predict(sess, g, input=obs, model_name=args.model)
                # print('cost:', np.mean(np.square(pred_-actions)))

    # Test prediction from saved model on an input to get an idea of whether the output is 
    # in the same range as the actual output.
    print('Predicting on a test input')
    test_input = obs[[-10], :]
    test_output = actions[[-10], :]
    pred_output = predict_action(input=test_input, model_name=args.model)
    print('Predicted output = ', pred_output)
    print('Actual output = ', test_output)



if __name__ == '__main__':
    main()






