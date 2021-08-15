#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K
import numpy as np
#  from .utils import _apply_weight_decays, _compute_eta_t
#  from .utils import _apply_lr_multiplier, _check_args, K_eval

import random
#  from termcolor import colored
'''Helper methods for optimizers
'''


def warn_str():
    #  return colored('WARNING: ', 'red')
    return ""


def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print(
                        (warn_str() + "{} l2-regularization = {} - should be "
                         "set 0 before compiling model").format(
                             weight_name, weight_l2))
    return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = _list_of_vals[idx]
    return _dict


def _get_layer_l2regs(layer):
    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_l2regs(layer)
    elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
        layer = layer.layer
    l2_lambda_kb = []
    for weight_name in ['kernel', 'bias']:
        _lambda = getattr(layer, weight_name + '_regularizer', None)
        if _lambda is not None:
            l2_lambda_kb.append(
                [getattr(layer, weight_name).name,
                 float(_lambda.l2)])
    return l2_lambda_kb


def _rnn_l2regs(layer):
    l2_lambda_krb = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            l2_lambda_krb += _cell_l2regs(layer.cell)
        return l2_lambda_krb
    else:
        return _cell_l2regs(layer.cell)


def _cell_l2regs(rnn_cell):
    cell = rnn_cell
    l2_lambda_krb = []  # kernel-recurrent-bias

    for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
        _lambda = getattr(cell, weight_type + '_regularizer', None)
        if _lambda is not None:
            weight_name = cell.weights[weight_idx].name
            l2_lambda_krb.append([weight_name, float(_lambda.l2)])
    return l2_lambda_krb


def _apply_weight_decays(cls, var, var_t):
    wd = cls.weight_decays[var.name]
    wd_normalized = wd * K.cast(
        K.sqrt(cls.batch_size / cls.total_iterations_wd), 'float32')
    var_t = var_t - cls.eta_t * wd_normalized * var

    if cls.init_verbose and not cls._init_notified:
        print('{} weight decay set for {}'.format(K_eval(wd_normalized),
                                                  var.name))
    return var_t


def _compute_eta_t(cls):
    PI = 3.141592653589793
    t_frac = K.cast(cls.t_cur / cls.total_iterations, 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
        (1 + K.cos(PI * t_frac))
    return eta_t


def _apply_lr_multiplier(cls, lr_t, var):
    multiplier_name = [
        mult_name for mult_name in cls.lr_multipliers if mult_name in var.name
    ]
    if multiplier_name != []:
        lr_mult = cls.lr_multipliers[multiplier_name[0]]
    else:
        lr_mult = 1
    lr_t = lr_t * lr_mult

    if cls.init_verbose and not cls._init_notified:
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
                '%.e' % K_eval(lr_t), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(
                var.name, K_eval(lr_t)))
    return lr_t


def _check_args(total_iterations, use_cosine_annealing, weight_decays):
    if use_cosine_annealing and total_iterations != 0:
        print('Using cosine annealing learning rates')
    elif (use_cosine_annealing
          or weight_decays != {}) and total_iterations == 0:
        print(warn_str() + "'total_iterations'==0, must be !=0 to use " +
              "cosine annealing and/or weight decays; " +
              "proceeding without either")


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")


def K_eval(x, backend=K):
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception as e:
        try:
            eval_fn = K.function([], [x])
            return eval_fn([])[0]
        except Exception as e:
            return K.eager(K.eval)(x)


class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".
        batch_size:       int >= 1. Train input batch size; used for normalization
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 amsgrad=False,
                 batch_size=32,
                 total_iterations=0,
                 total_iterations_wd=None,
                 use_cosine_annealing=False,
                 weight_decays=None,
                 lr_multipliers=None,
                 init_verbose=True,
                 eta_min=0,
                 eta_max=1,
                 t_cur=0,
                 **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate,
                                            name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.batch_size = K.variable(batch_size,
                                         dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.amsgrad = amsgrad
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay *
                             K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i))
            for (i, p) in enumerate(params)
        ]
        vs = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i))
            for (i, p) in enumerate(params)
        ]

        if self.amsgrad:
            vhats = [
                K.zeros(K.int_shape(p),
                        dtype=K.dtype(p),
                        name='vhat_' + str(i)) for (i, p) in enumerate(params)
            ]
        else:
            vhats = [
                K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))
            ]
        self.weights = [self.iterations] + ms + vs + vhats

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = lr_t * self.eta_t  # for external tracking

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Learning rate multipliers
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        self._init_notified = True
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K_eval(self.learning_rate)),
            'beta_1': float(K_eval(self.beta_1)),
            'beta_2': float(K_eval(self.beta_2)),
            'decay': float(K_eval(self.decay)),
            'batch_size': int(K_eval(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NadamW(Optimizer):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    # Arguments (other): see AdamW
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """
    def __init__(self,
                 learning_rate=0.002,
                 beta_1=0.9,
                 beta_2=0.999,
                 batch_size=32,
                 total_iterations=0,
                 total_iterations_wd=None,
                 use_cosine_annealing=False,
                 weight_decays=None,
                 lr_multipliers=None,
                 init_verbose=True,
                 eta_min=0,
                 eta_max=1,
                 t_cur=0,
                 **kwargs):
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(NadamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.learning_rate = K.variable(learning_rate,
                                            name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.batch_size = K.variable(batch_size,
                                         dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.use_cosine_annealing = use_cosine_annealing
        self.init_verbose = init_verbose

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (
            1. - 0.5 *
            (K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (
            1. - 0.5 * (K.pow(K.cast_to_floatx(0.96),
                              (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [
            K.zeros(shape, name='m_' + str(i))
            for (i, shape) in enumerate(shapes)
        ]
        vs = [
            K.zeros(shape, name='v_' + str(i))
            for (i, shape) in enumerate(shapes)
        ]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = self.learning_rate * self.eta_t  # for external tracking

        for p, g, m, v in zip(params, grads, ms, vs):
            # Learning rate multipliers
            lr_t = self.learning_rate
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p)

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (momentum_cache_t_1 *
                                                           m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            p_t = p - self.eta_t * lr_t * m_t_bar / (K.sqrt(v_t_prime) +
                                                     self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        self._init_notified = True
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include m_schedule at head of the weight list. Set
        # m_schedule to 1.
        if len(params) == len(weights) + 1:
            weights = [weights[0]] + [np.array(1.)] + weights[1:]
        super(NadamW, self).set_weights(weights)

    def get_config(self):
        config = {
            'learning_rate': float(K_eval(self.learning_rate)),
            'beta_1': float(K_eval(self.beta_1)),
            'beta_2': float(K_eval(self.beta_2)),
            'epsilon': self.epsilon,
            'schedule_decay': self.schedule_decay,
            'batch_size': int(K_eval(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        }
        base_config = super(NadamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGDW(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    # Arguments (other): see AdamW
    """
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.,
                 nesterov=False,
                 batch_size=32,
                 total_iterations=0,
                 total_iterations_wd=None,
                 use_cosine_annealing=False,
                 weight_decays=None,
                 lr_multipliers=None,
                 init_verbose=True,
                 eta_min=0,
                 eta_max=1,
                 t_cur=0,
                 **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate,
                                            name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.batch_size = K.variable(batch_size,
                                         dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.nesterov = nesterov
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay *
                             K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [
            K.zeros(shape, name='moment_' + str(i))
            for (i, shape) in enumerate(shapes)
        ]
        self.weights = [self.iterations] + moments

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = lr * self.eta_t  # for external tracking

        for p, g, m in zip(params, grads, moments):
            # Learning rate multipliers
            lr_t = self.learning_rate
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p)

            v = self.momentum * m - self.eta_t * lr_t * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                p_t = p + self.momentum * v - self.eta_t * lr_t * g
            else:
                p_t = p + v

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        self._init_notified = True
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K_eval(self.learning_rate)),
            'momentum': float(K_eval(self.momentum)),
            'decay': float(K_eval(self.decay)),
            'nesterov': self.nesterov,
            'batch_size': int(K_eval(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        }
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
