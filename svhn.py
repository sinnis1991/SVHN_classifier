# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep SVHN classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from extract_data import extract, extract_new
import argparse
import sys
import tempfile
import numpy as np

import tensorflow as tf
from ops import conv2d
FLAGS = None
epoch = 1
batch_size = 64
c_dim = 128
save_file = False
train_cnn = False
output_accuracy = True
checkpoint_path = "./checkpoint_SuperSuper/model.ckpt"
def deepnn(x,reuse=False):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.variable_scope('svhn_classifier') as scope:
    if reuse:
        scope.reuse_variables()

    with tf.name_scope('reshape'):
      x_im = tf.reshape(x, [-1, 32, 32, 3])

    with tf.name_scope('2float'):
      x_image = (tf.to_float(x_im)-127.5)/127.5

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    # with tf.name_scope('conv1'):
    #   W_conv1 = weight_variable([5, 5, 3, c_dim])
    #   b_conv1 = bias_variable([c_dim])
    #   h_conv1 = tf.nn.relu(conv2d_S(x_image, W_conv1) + b_conv1)

    h_conv1 = conv2d(x_image, c_dim,
               k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.1,
              name="conv1")

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
      h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('bn1') as scope:
      batch1 = tf.contrib.layers.batch_norm(h_pool1, decay=0.9, updates_collections=None, epsilon=1e-5,
                                   center=True, scale=True, is_training=True,scope=scope)

    with tf.name_scope('relu1'):
      relu1 = lrelu_S(batch1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    # with tf.name_scope('conv2'):
    #   W_conv2 = weight_variable([5, 5, c_dim, c_dim*2])
    #   b_conv2 = bias_variable([c_dim*2])
    #   h_conv2 = tf.nn.relu(conv2d_S(relu1, W_conv2) + b_conv2)
    h_conv2 = conv2d(relu1, c_dim*2,
                     k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.1,
                     name="conv2")

    # Second pooling layer.
    with tf.name_scope('pool2'):
      h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('bn2') as scope:
      batch2 = tf.contrib.layers.batch_norm(h_pool2, decay=0.9, updates_collections=None, epsilon=1e-5,
                                            center=True, scale=True, is_training=True,scope=scope)
    with tf.name_scope('relu2'):
      relu2 = lrelu_S(batch2)

    # Third convolutional layer -- maps 64 feature maps to 128.
    # with tf.name_scope('conv3'):
    #   W_conv3 = weight_variable([5, 5, c_dim*2, c_dim*2*2])
    #   b_conv3 = bias_variable([c_dim*2*2])
    #   h_conv3 = tf.nn.relu(conv2d_S(relu2, W_conv3) + b_conv3)
    h_conv3 = conv2d(relu2, c_dim*2*2,
                     k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.1,
                     name="conv3")


    # Third pooling layer.
    with tf.name_scope('pool3'):
      h_pool3 = max_pool_2x2(h_conv3)

    with tf.variable_scope('bn3') as scope:
      batch3 = tf.contrib.layers.batch_norm(h_pool3, decay=0.9, updates_collections=None, epsilon=1e-5,
                                            center=True, scale=True, is_training=True,scope=scope)
    with tf.name_scope('relu3'):
      relu3 = lrelu_S(batch3)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.variable_scope('fc1') as scope:
      # W_fc1 = weight_variable([4 * 4 * c_dim*2*2, 1024])
      # b_fc1 = bias_variable([1024])
      w = tf.get_variable('w', [4 * 4 * c_dim*2*2, 1024],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))
      biases = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.1))

      h_pool2_flat = tf.reshape(relu3, [-1, 4*4*c_dim*2*2])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w) + biases)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropoutX'):
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.variable_scope('fc2') as scope:
      # W_fc2 = weight_variable([1024, 10])
      # b_fc2 = bias_variable([10])

      w2 = tf.get_variable('w', [1024,10],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))
      biases2 = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.1))

      y_conv = tf.matmul(h_fc1_drop, w2) + biases2
    return y_conv, keep_prob


def conv2d_S(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def lrelu_S(x, leak=0.2):
  f1 = 0.5 * (1 + leak)
  f2 = 0.5 * (1 - leak)
  return f1 * x + f2 * abs(x)

def main(_):

    im, label = extract_new('./','full_train_data.mat')
    im2, label2 = extract('./', 'test_32x32.mat')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 32,32,3])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                              logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
        print ("Model restored.")
        print ("All trainable variables:")
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)
        for var, val in zip(tvars, tvars_vals):
            print(var.name)


        batch_num = 1+(im.shape[0] // batch_size)
        show_accuracy_interval = batch_num//50

        for i in range(epoch):

          for j in range(batch_num):

            feed_im = im[batch_size*j:min(batch_size*(j+1),im.shape[0])]
            feed_label = label[batch_size*j:min(batch_size*(j+1),im.shape[0])]

            if j % show_accuracy_interval == 0 or j == batch_num-1:
              train_accuracy = accuracy.eval(feed_dict={
                x: feed_im, y_: feed_label, keep_prob: 1.0})
              print('epoch %d batch (%d/%d), training accuracy %g' % (i,j,batch_num, train_accuracy))
              if train_cnn:
                train_step.run(feed_dict={x: feed_im, y_: feed_label, keep_prob: 0.5})

          if output_accuracy :
            accuracy_sum=0
            for k in range(im2.shape[0]//batch_size):
              accuracy_sum += accuracy.eval(feed_dict={
                x: im2[batch_size*k:batch_size*(k+1)],
                y_: label2[batch_size*k:batch_size*(k+1)],
                keep_prob: 1.0})
            print('epoch %d test accuracy %g' % (i,accuracy_sum/(im2.shape[0]//batch_size)))

        if save_file:
          save_path = saver.save(sess, checkpoint_path)
          print("Model saved in file: %s" % save_path)

        # accuracy= accuracy.eval(feed_dict={
        #   x: im2[:64],
        #   y_: label2[:64],
        #   keep_prob: 1.0})
        #
        # print('epoch %d test accuracy %g' % (i, accuracy))

  # vars = tf.contrib.framework.list_variables('./checkpoint/model.ckpt')
  # with tf.Graph().as_default(), tf.Session().as_default() as sess:
  #   new_vars = []
  #   for name, shape in vars:
  #     v = tf.contrib.framework.load_variable('./checkpoint/model.ckpt', name)
  #     new_vars.append(tf.Variable(v, name=name.replace('BatchNorm', 'svhn_classifier/BatchNorm')))
  #
  #   saver = tf.train.Saver(new_vars)
  #   sess.run(tf.global_variables_initializer())
  #   saver.save(sess, './checkpoint_new/model.ckpt')
  # for var in new_vars:
  #   print(var.name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)