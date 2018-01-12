"""Module for creating architecture
TODO:
  * Implement dot dataset
  * Implement the egomotion dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tfdeploy as td
import numpy as np

import options
import utils

slim = tf.contrib.slim

# This architecture as of now is not flexible - you cannot just
# specify a number of layers and the architecture will fiture itself out
# TODO: Make flexible
def network(x, opts, regularizer=None, debug=False):
  """Build network architecture."""
  if debug:
    pr = utils.myprint
  else:
    pr = utils.noop
  layers = [x]
  nlayers = opts.architecture['nlayers']
  img_size = opts.net_img_size
  kernel_sizes = opts.architecture['kernel_sizes']
  filters = opts.architecture['filters']
  paddings = opts.architecture['paddings']
  activation = opts.architecture['activation']
  out_sizes = [ img_size // 2**i for i in range(nlayers+1) ]
  with tf.name_scope('network'):
    with tf.name_scope('downsampling'):
      pr("Starting downsampling...")
      for i in range(nlayers):
        conv = tf.layers.conv2d(inputs=layers[-1],
                                filters=filters[i],
                                kernel_size=kernel_sizes[i],
                                strides=1,
                                padding=paddings[i],
                                activation=activation,
                                kernel_regularizer=regularizer)
        pool = tf.layers.max_pooling2d(conv,
                                       pool_size=[2, 2],
                                       strides=[2, 2])
        layers.append(pool)

    # 'Linear' part
    with tf.name_scope('fully_connected'):
      pr("Fully connected...")
      conv = tf.layers.conv2d(inputs=layers[-1],
                              filters=opts.architecture['embedding_size'],
                              kernel_size=out_sizes[-1],
                              strides=1,
                              padding='valid',
                              activation=activation,
                              kernel_regularizer=regularizer)
      layers.append(conv)
    
    # Upsampling
    with tf.name_scope('upsampling'):
      pr("Starting upsampling...")
      upsamp = tf.image.resize_bilinear(layers[-1],
                                       size=[out_sizes[nlayers]]*2)
      for i in range(nlayers):
        j = nlayers-1-i
        concat = tf.concat([upsamp, layers[j+1]], 3)
        # layers.append(concat)
        conv = tf.layers.conv2d(inputs=concat,
                                filters=2*filters[j],
                                kernel_size=kernel_sizes[j],
                                strides=1,
                                padding='valid',
                                activation=activation,
                                kernel_regularizer=regularizer)
        # layers.append(conv)
        upsamp = tf.image.resize_bilinear(conv,
                                          size=[out_sizes[j]]*2)
        layers.append(upsamp)
    with tf.name_scope('soft_max'):
      conv = tf.layers.conv2d(inputs=layers[-1],
                              filters=opts.nclasses,
                              kernel_size=1,
                              strides=1,
                              padding='valid',
                              activation=None,
                              kernel_regularizer=regularizer)
      layers.append(conv)
      # softmax = tf.contrib.layers.softmax(conv)
      # layers.append(softmax)
     

    for i, l in enumerate(layers):
      pr("{}: {}".format(i, l))
    return layers

def build_architecture(opts, sample):
  opts.architecture = {}
  opts.architecture['nlayers'] = 6
  opts.architecture['kernel_sizes'] = [ 5, 5, 3, 3, 3, 3 ]
  opts.architecture['filters'] = [ 64, 128, 256, 256, 512, 1024 ]
  opts.architecture['paddings'] = [ 'same' ] * 7 
  opts.architecture['activation'] = utils.activation(opts.activation_type)
  opts.architecture['embedding_size'] = 2048
  return network(sample['image'], opts, regularizer=None)
  

if __name__ == '__main__':
  opts = options.get_opts()
  opts.architecture = {}
  opts.architecture['nlayers'] = 6
  opts.architecture['kernel_sizes'] = [ 5, 5, 3, 3, 3, 3 ]
  opts.architecture['filters'] = [ 64, 128, 256, 256, 512, 1024 ]
  opts.architecture['paddings'] = [ 'same' ] * 7 
  opts.architecture['activation'] = utils.activation(opts.activation_type)
  opts.architecture['embedding_size'] = 2048
  size = [None, opts.net_img_size, opts.net_img_size, opts.nchannels]
  x = tf.placeholder(tf.float32, size)
  network(x, opts, debug=True)



