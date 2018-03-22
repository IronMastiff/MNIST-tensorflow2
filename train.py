import tensorflow as tf
import numpy as np
import argparse
import os

import Utils

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'MNIST_data', one_hot = True )

'''--------Fit datas--------'''
datas_train = mnist.train.images      # ( 55000, 784 )
train_labels = mnist.train.labels    #( 55000, 10 )
datas_test = mnist.test.images
test_labels = mnist.test.labels

'''--------读取config中的配置--------'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default = 'config/config.yml', help = 'the path to the congig file' )
    return parser.parse_args()

def main( FLAGS ):
    mini_batches = Utils.random_mini_batches( datas_train, train_labels, FLAGS.batch_size )
    ( get_shape1, get_shape2 ) = mini_batches[0]    # ( 32, 784 ) ( 32, 10 )

    datas, labels = Utils.creat_placeholders( get_shape1, get_shape2 )

    layer_output = Utils.net( datas )

    cost = Utils.compute_cost( layer_output, labels )
    tf.summary.scalar( 'cost', cost )

    optimizer = tf.train.AdamOptimizer( learning_rate = FLAGS.learning_rate ).minimize( cost )

    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()

        last_checkpoint = tf.train.latest_checkpoint( 'checkpoint' )
        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
            print( 'reuse the model' )
        else: sess.run( init )

        writer = tf.summary.FileWriter( "logs/", sess.graph )

        for epoch in range( FLAGS.epoch ):
            epoch_cost = 0
            for mini_batch in mini_batches:
                ( minibatch_X, minibatch_Y ) = mini_batch

                _, mini_batch_cost = sess.run( [optimizer, cost], feed_dict = {datas: minibatch_X, labels: minibatch_Y} )

                epoch_cost += mini_batch_cost / get_shape1.shape[0]

            if epoch % 100 == 0:
                print ( "Cost after epoch %i: %f" % ( epoch, epoch_cost ) )

            if epoch % 5 == 0:
                rs = sess.run( merged, feed_dict={datas: minibatch_X, labels: minibatch_Y} )
                writer.add_summary( rs, epoch )

        saver.save( sess, 'checkpoint/save_net.ckpt' )

        correct_prediction = tf.equal( tf.argmax( layer_output, 1 ), tf.argmax( labels, 1 ) )

        accuracy = tf.reduce_mean( tf.cast( correct_prediction, "float" ) )

        print( "Train Accuracy:", accuracy.eval( {datas: datas_train, labels: train_labels} ) )
        print( "Test Accuracy:", accuracy.eval( {datas: datas_test, labels: test_labels} ) )



if __name__ == '__main__':
    # x = np.reshape( mnist.train.labels,( mnist.train.labels.shape[0], 1 ) )
    # print( mnist.train.labels[:1])

    args = parse_args()
    FLAGS = Utils.read_config_file( args.conf )
    main( FLAGS )
