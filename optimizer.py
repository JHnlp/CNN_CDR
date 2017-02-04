# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
import numpy as np
from keras import backend as K


def rectify(X):
    return T.maximum(X, 0.)
    # return T.nnet.relu(X, 0.)  # this doesn't work, I don't know why?


def negative_log_likelihood(p_y_given_x, y):
    """Return the mean of the negative log-likelihood of the prediction
    of this model under a given target distribution.

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label

    Note: we use the mean instead of the sum so that
          the learning rate is less dependent on the batch size
    """
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
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])


def errors(y_pred, y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label
    """

    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', y_pred.type))
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y))
    else:
        raise NotImplementedError()


def sgd(loss, params, lr=0.01):
    """
    input:
        cost: cost function
        params: parameters
        lr: learning rate
    output:
        update rules
    """
    grads = T.grad(cost=loss, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def RMSprop(loss, params, lr=0.001, rho=0.9, epsilon=1e-8):
    grads = T.grad(cost=loss, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def adagrad_update(loss, params, lr=0.01, epsilon=1e-8):
    params = params
    accumulators = [theano.shared(np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX))
                    for p in params]
    grads = [T.grad(loss, param) for param in params]
    grads = grads
    updates = []
    for param, g, acc in zip(params, grads, accumulators):
        new_acc = acc + g ** 2
        updates.append((acc, new_acc))
        updates.append((param, param - lr * g / T.sqrt(new_acc + epsilon)))
    return updates


def adadelta(loss, params, lr=0.4, rho=0.95, epsilon=1e-8):
    grads = T.grad(cost=loss, wrt=params)
    accumulators = [theano.shared(value=np.zeros(p.get_value().shape, dtype=theano.config.floatX))
                    for p in params]
    delta_accumulators = [theano.shared(value=np.zeros(p.get_value().shape, dtype=theano.config.floatX))
                          for p in params]
    updates = []
    for p, g, acc, d_acc in zip(params, grads, accumulators, delta_accumulators):
        # update accumulator
        new_acc = rho * acc + (1. - rho) * T.sqr(g)
        updates.append((acc, new_acc))

        # use the new accumulator and the *old* delta_accumulator
        update = g * T.sqrt(d_acc + epsilon) / T.sqrt(new_acc + epsilon)
        new_p = p - lr * update
        updates.append((p, new_p))

        # update delta_accumulator
        new_d_a = rho * d_acc + (1 - rho) * T.square(update)
        updates.append((d_acc, new_d_a))
    return updates


def momentum_grad(loss, params, lr=0.3, momentum=0.95):
    ''' Momentum GD with gradient clipping. '''
    grads = T.grad(loss, params)
    momentum_velocity_ = [0.] * len(grads)
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    updates = OrderedDict()
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    scaling_den = T.maximum(5.0, grad_norm)

    for n, (param, grads) in enumerate(zip(params, grads)):
        grads = T.switch(not_finite, 0.1 * param, grads * (0.5 / scaling_den))
        velocity = momentum_velocity_[n]
        update_step = momentum * velocity - lr * grads
        momentum_velocity_[n] = update_step
        updates[param] = param + update_step

    return updates


def adagrad(loss, params, lr=0.001, momentum=0.95, epsilon=1e-8):
    accumulators = [theano.shared(np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX))
                    for p in params]
    grads = [T.grad(loss, param) for param in params]

    updates = []
    for param, grad, acc in zip(params, grads, accumulators):
        new_acc = acc + grad ** 2
        new_acc_sqrt = T.sqrt(new_acc + epsilon)
        grad = grad / new_acc_sqrt

        #       isinf = T.or_(T.isinf(grad), T.isnan(grad))
        #       grad = T.switch(isinf, 0., grad)

        updates.append((acc, new_acc))
        updates.append((param, param - (lr * (grad + momentum * param))))

    return updates
