import os
import numpy as np
import math
import tensorflow as tf
import yaml

def random_mini_batches( X, Y, random_mini_batches_size ):
    size = X.shape[0]
    mini_batches = []

    permutation = list( np.random.permutation( size ) )
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    num_complitete_minibatches = math.floor( size / random_mini_batches_size )
    for i in range( 0, num_complitete_minibatches, random_mini_batches_size ):
        mini_batch_X = shuffled_X[ i * random_mini_batches_size : ( i + 1 ) * random_mini_batches_size, :]
        mini_batch_Y = shuffled_Y[ i * random_mini_batches_size : ( i + 1 ) * random_mini_batches_size, :]
        mini_batch = ( mini_batch_X, mini_batch_Y )
        mini_batches.append( mini_batch )

    if size % random_mini_batches_size != 0:
        mini_batch_X = shuffled_X[num_complitete_minibatches * random_mini_batches_size : size, :]
        mini_batch_Y = shuffled_Y[num_complitete_minibatches * random_mini_batches_size : size, :]

        mini_batch = ( mini_batch_X, mini_batch_Y )
        mini_batches.append( mini_batch )

    return mini_batches

'''--------Creat placeholder--------'''
def creat_placeholders( datas, labels ):
    X = tf.placeholder( tf.float32, [None, datas.shape[1]] )
    Y = tf.placeholder( tf.float32, [None, labels.shape[1]] )

    return X, Y

'''---------The net--------'''
def net( input ):
    layer1 = tf.layers.dense( input, 1024, activation = tf.nn.relu,
                              kernel_initializer = tf.contrib.layers.xavier_initializer() )
    layer2 = tf.layers.dense( layer1, 512, activation = tf.nn.relu,
                              kernel_initializer = tf.contrib.layers.xavier_initializer() )
    layer3 = tf.layers.dense( layer2, 10, activation = None,
                              kernel_initializer = tf.contrib.layers.xavier_initializer() )

    return layer3

class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def read_config_file( config_file ):
    with open( config_file ) as f:
        FLAGS = Flag( **yaml.load( f ) )
    return FLAGS

def compute_cost( input, labels ):
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = input, labels = labels ) )

    return cost