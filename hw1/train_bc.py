#!/usr/bin/env python

"""
Code to load an expert policy generate data and train a model via behavioral cloning.
Example usage:
    python train_bc.py expert-data-experts.RoboschoolAnt-v1.pkl \ 
           --model experts.RoboschoolAnt-v1 --train
"""


import pickle
import tensorflow as tf
import numpy as np
import itertools
import argparse
import random
from scipy import stats
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd


LEARNING_RATE = 0.001
NUM_ITERATIONS = 1000
N_HIDDEN_1 = 16
N_HIDDEN_2 = 16


def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1, name='hidden1')
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2, name='hidden2')
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='pred_action')
    return out_layer


def weights_and_biases(input_size, output_size):
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.zeros([input_size, N_HIDDEN_1]), name='weights1'),
        'h2': tf.Variable(tf.zeros([N_HIDDEN_1, N_HIDDEN_2]), name='weights2'),
        'out': tf.Variable(tf.zeros([N_HIDDEN_2, output_size]), name='weights_out')
    }
    biases = {
        'b1': tf.Variable(tf.zeros([N_HIDDEN_1]), name='biases1'),
        'b2': tf.Variable(tf.zeros([N_HIDDEN_2]), name='biases2'),
        'out': tf.Variable(tf.zeros([output_size]), name='biases_out')
    }
    return weights, biases


def linear_model(x, input_size, output_size):
    weights = tf.Variable(tf.zeros([input_size, output_size]), name='weights')
    biases = tf.Variable(tf.zeros([output_size]), name='biases')
    out_layer = tf.add(tf.matmul(x, weights), biases, name='pred_action')
    return out_layer, weights, biases


def setup_graph(input_size, output_size, model_type_mlp=False):
    x = tf.placeholder(tf.float32, [None, input_size], name='obs')
    y = tf.placeholder(tf.float32, [None, output_size], name='action')

    # Construct a MLP
    if model_type_mlp:
        weights, biases = weights_and_biases(input_size, output_size)
        pred = multilayer_perceptron(x, weights, biases)
    else:
        pred, weights, biases = linear_model(x, input_size, output_size)
    # Mean squared error
    cost = tf.reduce_mean(tf.pow(pred-y, 2))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    if model_type_mlp:
        return {'pred':pred, 'cost': cost, 'train_op':optimizer, 'input':x, 'output':y, 'weights':weights['h1'], 'biases':biases['b1']}
    else:
        return {'pred':pred, 'cost': cost, 'train_op':optimizer, 'input':x, 'output':y, 'weights':weights, 'biases':biases}
    

def train(sess, g, input, output, model_name):
    costs = []
    preds = []
    val_preds = []
    val_costs = []
    val_size = round((input.shape[0] * 8) / 10)
    train_input = input[:val_size, :]
    train_output = output[:val_size, :]
    val_input = input[val_size:, :]
    val_output = output[val_size:, :]
    for i in range(NUM_ITERATIONS):
        pred_, cost_, _, weights_ = sess.run(
            [g['pred'], g['cost'], g['train_op'], g['weights']], 
            feed_dict={g['input']:train_input, g['output']:train_output})
        val_pred_, val_cost_ = sess.run(
            [g['pred'], g['cost']], 
            feed_dict={g['input']:val_input, g['output']:val_output})

        costs.append(cost_)
        preds.append(pred_)
        val_preds.append(val_pred_)
        val_costs.append(val_cost_)
        if i % 10 == 0:
            print("Iteration", i, "Train Cost", cost_, "Val Cost", val_cost_)
    print("pred_ shape", pred_.shape)
    print('Train Input stats\n', pd.DataFrame(train_input).describe())
    print('Preds stats\n', pd.DataFrame(pred_).describe())
    print('Output stats\n', pd.DataFrame(train_output).describe())
    print('Val Preds stats\n', pd.DataFrame(val_pred_).describe())
    print('Val Output stats\n', pd.DataFrame(train_output).describe())
    print('Learnt Weights\n', pd.DataFrame(weights_).describe())
    # Save model to disk.
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # output_filename = model_name + '_' + 'model.ckpt' 
    saved_path = saver.save(sess, model_name)
    print("Model saved:", model_name, ' at ', saved_path)

    # Plot all results
    plt.figure()
    plt.plot(list(range(NUM_ITERATIONS)), costs)
    plt.plot(list(range(NUM_ITERATIONS)), val_costs, 'r')
    #plt.legend()
    plt.title('Cost vs Iteration num')
    plt.show()


def predict(sess, g, input, model_name):
    #saver = tf.train.Saver()
    #output_filename = model_name + '_' + 'model.ckpt' 
    saver = tf.train.import_meta_graph(model_name + '.meta')
    saver.restore(sess, model_name)
    print("Model restored:", model_name)
    pred_ = sess.run([g['pred']], feed_dict={g['input']:input})
    return pred_


def predict_action(input, model_name):
    # define the tensorflow graph
    tf.reset_default_graph()

    with tf.Graph().as_default():
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
    parser.add_argument('data_filename', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_type_mlp', action='store_true')
   
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
                train(sess,g,input=obs, output=actions, model_name=args.model)

                # Test that saved model works as intended
                pred_ = predict(sess, g, input=obs, model_name=args.model)
                print('cost:', np.mean(np.square(pred_-actions)))
            """
            num = 4
            plt.figure()
            xdata = np.array(range(0, num))
            preds1 = np.array(preds)
            print(preds1.shape, preds1[:,0:num,0].shape)
            sns.tsplot(time=xdata, data=preds1[:,0:num,0], color='r', linestyle='-')
            sns.tsplot(time=xdata, data=preds1[:,0:num,1], color='b', linestyle='--')
            sns.tsplot(time=xdata, data=preds1[:,0:num,2], color='g', linestyle='-.')
            sns.tsplot(time=xdata, data=preds1[:,0:num,3], color='k', linestyle=':')

            plt.ylabel('Action', fontsize=25)
            plt.xlabel('Observation num', fontsize=25, labelpad=-4)
            plt.title('Robot performance', fontsize=25)
            plt.legend(loc='bottom left')
            plt.show()
            """

    ## Test prediction from saved model on actual output
    print('Predicting on a test input')
    test_input = obs[[-10], :]
    test_output = actions[[-10], :]
    pred_output = predict_action(input=test_input, model_name=args.model)
    print('Predicted output = ', pred_output)
    print('Actual output = ', test_output)


if __name__ == '__main__':
    main()






