import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

import Utils

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'MNIST_data', one_hot = True )

'''--------读取config文件中的数据--------'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default = 'config/config.yml', help = 'the path to the congig file' )
    return parser.parse_args()

'''--------Fit datas--------'''
datas_train = mnist.train.images
train_labels = mnist.train.labels
datas_test = mnist.test.images
test_labels = mnist.test.labels

last_checkpoint = tf.train.latest_checkpoint( 'checkpoint' )

def main( FLAG ):
    datas, labels = Utils.creat_placeholders( datas_test, test_labels )

    layer_output = Utils.net( datas )

    image = datas_test[FLAGS.index, :]
    image = image.reshape( 28, -1 )
    plt.imshow( image )
    plt.show()
    print( 'The real label: ', np.argmax( test_labels[FLAGS.index, :] ) )

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
            print( 'reuse the model' )
        else:
            print( "net hadn't be trand" )

        prediction = tf.argmax( layer_output[FLAGS.index, :] )
        print( 'Machine prediced: ', prediction.eval( {datas : datas_test, labels : test_labels} ) )

if __name__ == '__main__':
    # x = np.reshape( mnist.train.labels,( mnist.train.labels.shape[0], 1 ) )
    # print( mnist.train.labels[:1])

    args = parse_args()
    FLAGS = Utils.read_config_file( args.conf )
    main( FLAGS )