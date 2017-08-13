#! /usr/bin/ev python

import tensorflow as tf


class ActionPredictor():
    def __init__(self, model_name, input_tensor_name="obs:0", output_tensor_name="pred_action:0"):
        self._model_name = model_name
        self._input_tensor_name = input_tensor_name
        self._output_tensor_name = output_tensor_name

    def start(self):
        """ Starts the graph and the session."""
        print('***************Starting ActionPredictor graph: ', self._model_name, tf.train.latest_checkpoint('./'))
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        # Start the session
        self.sess = tf.Session(graph=graph)
        saver = tf.train.import_meta_graph(self._model_name + '.meta')
        self.input_tensor = graph.get_tensor_by_name(self._input_tensor_name)
        self.pred_tensor = graph.get_tensor_by_name(self._output_tensor_name)
        init = tf.global_variables_initializer()

        self.sess.run(init)
        saver.restore(self.sess, self._model_name)
        print('Loaded checkpoint from ', self._model_name)


    def predict_action(self, input):
                # print("Model restored:", model_name)
        pred_ = self.sess.run(self.pred_tensor, feed_dict={self.input_tensor: input})
        #print("pred=", pred_[0])
        #print(pred_[0].shape)
        return pred_


    def stop(self):
        self.sess.close()
 
