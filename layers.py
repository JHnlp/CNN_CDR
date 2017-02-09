# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import numpy as np
import os
import sys
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from gensim.models import *


class SoftMaxLayer(object):
    """SoftMaxLayer, i.e. Multi-class Logistic Regression Class.
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyperplane-k
        # p_y_given_x is a matrix
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred_prob = T.max(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def accuracy(self, y):
        err = self.errors(y)
        return 1. - err


class EntityLookUpTableLayer(object):
    def __init__(self, inputs, weights):
        """
        :param inputs: matrix of entity words
        :param weights:
        """
        if type(inputs) is list:
            self.inputs = inputs
        else:
            self.inputs = [inputs]

        self.W = weights

        self.outputs = []

        for ent_input in self.inputs:
            results, _ = theano.scan(fn=lambda x: T.mean(self.W[x[(x > -1.).nonzero()]], axis=0),
                                     sequences=[ent_input])
            self.outputs.append(results)

        self.params = [self.W]


class Seq2VecLookUpTableLayer(object):
    def __init__(self, inputs, vocab_size=None, dim_size=None, weights=None, scale=0.05, name=None):
        """
        :param inputs: batch of words sequence
        :param weights:
        """
        if type(inputs) is list:
            self.inputs = inputs
        else:
            self.inputs = [inputs]

        if weights:
            self.W = weights
        else:
            if not (vocab_size and dim_size):
                raise Exception('W cannot be initialized!')
            else:
                #np.random.seed(123456)
                self.W = theano.shared(
                    np.random.uniform(low=-scale, high=scale, size=(vocab_size, dim_size)).astype('float32'), name=name)

        self.outputs = []
        for seq in self.inputs:
            results, _ = theano.scan(fn=lambda x: self.W[x], sequences=[seq])
            self.outputs.append(results)

        self.params = [self.W]


class Path2VecLookUpTableLayer(object):
    def __init__(self, inputs, vocab_size=None, dim_size=None, weights=None, scale=0.05, name=None):
        """
        :param inputs: 3-D tensor
        :param weights:
        """
        if type(inputs) is list:
            self.inputs = inputs
        else:
            self.inputs = [inputs]

        if weights:
            self.W = weights
        else:
            if not (vocab_size and dim_size):
                raise Exception('W cannot be initialized!')
            else:
                self.W = theano.shared(
                    np.random.uniform(low=-scale, high=scale, size=(vocab_size, dim_size)).astype('float32'), name=name)

        self.outputs = []

        def lookup(tensor_2D):
            results, _ = theano.scan(fn=lambda x: self.W[x], sequences=[tensor_2D])
            return results

        for tensor_3D in self.inputs:
            results, _ = theano.scan(fn=lookup, sequences=[tensor_3D])
            self.outputs.append(results)
        self.params = [self.W]


class HiddenLayer(object):
    def __init__(self, inputs, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh
        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type inputs: theano.tensor.dmatrix
        :param inputs: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = inputs

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        rng = np.random.RandomState(2345347)
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]


def dropout_tensor(tensor, p=0.1):
    """

    :param tensor:
    :param p: is the probability of dropping a unit
    :return:
    """
    srng = RandomStreams(123455946)
    mask = srng.binomial(n=1, p=1 - p, size=tensor.shape)

    output = tensor * T.cast(mask, theano.config.floatX)
    return output


def dropout(tensor, n_in, p=0.2):
    rng = np.random.RandomState()
    mask = rng.binomial(n=1, p=1 - p, size=(n_in,))
    output = tensor * T.cast(mask, theano.config.floatX)

    return output


class DropoutLayer(object):
    def __init__(self, input, n_in):
        self.input = input

        rng = np.random.RandomState()
        p = 0.2
        mask = rng.binomial(n=1, p=1 - p, size=(n_in,))
        self.output = self.input * T.cast(mask, theano.config.floatX)


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, filter_shape, input_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter rows, filter cols)

        :type input_shape: tuple or list of length 4
        :param input_shape: (batch size, num input feature maps, image rows, image cols)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        rng = np.random.RandomState()

        assert input_shape[1] == filter_shape[1]
        self.input = input

        self.filter_shape = filter_shape

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(input=input, filters=self.W, filter_shape=filter_shape
                          , input_shape=input_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class MatrixConvPoolLayer(object):
    def __init__(self, input, sent_position_tags, nb_filters, window_size, embd_dim, activation=T.tanh):
        """

        :param input: 3-D tensor, i.e. (batch_size, sentence_length, word_embedding_dim)
        :param sent_position_tags: the actual first and last words position in padded sentence
        :param nb_filters:
        :param window_size: left-current-right window size
        :param embedding_dim:
        :param activation:
        """
        self.input = input
        self.position_tags = sent_position_tags

        rng = np.random.RandomState()
        W_bound = np.sqrt(6. / (nb_filters + window_size * embd_dim)).astype('float32')
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(nb_filters, window_size * embd_dim)),
                              dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        def mat_conv(sent_matrix, first_last_pos, W):
            orginal_sent = sent_matrix

            half_window_size = window_size // 2
            mini_window = range(-half_window_size, half_window_size + 1)
            left = theano.shared(np.zeros((half_window_size, embd_dim), dtype=theano.config.floatX))
            right = theano.shared(np.zeros((half_window_size, embd_dim), dtype=theano.config.floatX))

            new_padded = T.join(0, left, orginal_sent, right)

            def tiny_concat(*V):
                return T.join(0, *V)

            rlts, _ = theano.scan(fn=tiny_concat, sequences=dict(input=new_padded, taps=mini_window))
            tsr = T.max(T.dot(W, T.transpose(rlts)), axis=1)  # conv-and-pooling
            return tsr

        results, _ = theano.scan(fn=mat_conv, sequences=[self.input, self.position_tags], non_sequences=self.W)
        # f = theano.function(inputs=[X, P], outputs=results, allow_input_downcast=True, on_unused_input='warn')

        self.output = T.tanh(results)
        self.params = [self.W]


class SimpleConvPoolLayer(object):
    def __init__(self, input, filter_shape, activation=T.tanh):
        """

        :param input: 3-D tensor, i.e. (batch_size, sentence_length, word_embedding_dim)
        :param filter_shape: a 2-d matrix
        :param activation:
        """
        self.input = input

        rng = np.random.RandomState()
        W_bound = np.sqrt(6. / (filter_shape[0] + filter_shape[1])).astype('float32')
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(filter_shape[0], filter_shape[1])),
                              dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        results, _ = theano.scan(fn=lambda s, w, b: T.max(T.dot(w, T.transpose(s)), axis=1) + b,
                                 sequences=[self.input], non_sequences=[self.W, self.b])

        self.output = T.tanh(results)
        self.params = [self.W, self.b]


class MaxMinMeanConvPoolLayer(object):
    def __init__(self, input, filter_shape, activation=T.tanh):
        """

        :param input: 3-D tensor, i.e. (batch_size, sentence_length, word_embedding_dim)
        :param filter_shape: a 2-d matrix
        :param activation:
        """
        self.input = input

        rng = np.random.RandomState()
        W_bound = np.sqrt(6. / (filter_shape[0] + filter_shape[1])).astype('float32')
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(filter_shape[0], filter_shape[1])),
                              dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        results, _ = theano.scan(fn=lambda s, w, b: T.concatenate(
            [T.max(T.dot(w, T.transpose(s)), axis=1) + b, T.min(T.dot(w, T.transpose(s)), axis=1) + b,
             T.mean(T.dot(w, T.transpose(s)), axis=1) + b]), sequences=[self.input], non_sequences=[self.W, self.b])

        self.output = T.tanh(results)
        self.params = [self.W, self.b]
