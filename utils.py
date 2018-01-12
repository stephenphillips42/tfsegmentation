#!/usr/bin/python

import tensorflow as tf
from operator import mul

# Useful functions
def noop(x):
  pass

def myprint(x):
  print("DEBUG: {}".format(x))

# Augmentation utils
def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])


# Activation utils
def leakyRelu(x):
  alpha=0.01
  with tf.name_scope('leakyRelu'):
    return tf.maximum(x,tf.mul(x,alpha))

def relu_square(x):
  with tf.name_scope('relu_square'):
    return tf.pow(tf.maximum(x, 0.0),2.0)

def activation(activationType):
  if activationType == 'relu':
    return tf.nn.relu
  elif activationType == 'leakyrelu':
    return leakyRelu
  elif activationType == 'tanh':
    return tf.tanh
  elif activationType == 'relusq':
    return relu_square

# TODO: add deconvolution layer
# Layer utils
def fully_connected_layer(inputs, mid_size, out_size, activation, regularizer=None):
  """Default caffe style MRSA Fully Connected Layer. Assumed x.get_shape(0) == None."""
  with tf.name_scope('fc_layer'): 
    layer = tf.layers.dense(inputs=inputs,
                            units=mid_size,
                            activation=activation,
                            kernel_regularizer=regularizer)
    return tf.layers.dense(inputs=layer, units=out_size, kernel_regularizer=regularizer)

def linear_layer(x, mid_size, out_size, regularizer=None):
  """Default caffe style MRSA Fully Connected Layer. Assumed x.get_shape(0) == None."""
  with tf.name_scope('linear_layer'): 
    in_size = reduce(mul, x.get_shape()[1:].as_list(), 1)
    x2 = tf.reshape(x,[-1,in_size])
    return tf.layers.dense(inputs=x2, units=outSize, kernel_regularizer=regularizer)

def resnet_conv(inputs,
                filters,
                kernel_size,
                strides=(1,1),
                activation=None,
                batch_norm=True,
                kernel_regularizer=None):
  with tf.variable_scope('resconv_1'):
    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=filters[0],
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='SAME',
                             activation=activation,
                             kernel_regularizer=kernel_regularizer)
    if batch_norm:
      conv1 = tf.layers.batch_normalization(conv1)
  with tf.variable_scope('resconv_2'):
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=filters[1],
                             kernel_size=kernel_size,
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_regularizer=kernel_regularizer)
    if batch_norm:
      conv2 = tf.layers.batch_normalization(conv2)
  with tf.variable_scope('resconv_skip'):
    skip = tf.layers.conv2d(inputs=inputs,
                            filters=filters[1],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='SAME',
                            activation=None,
                            kernel_regularizer=kernel_regularizer)
    if batch_norm:
      skip = tf.layers.batch_normalization(skip)
  with tf.variable_scope('resconv_output'):
    output = activation(conv2 + skip)
  return output



