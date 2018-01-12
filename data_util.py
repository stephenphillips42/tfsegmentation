"""Module for generating or reorganizing various datasets.
This generates various datasets (dot, egomotion, etc.) and converts them to tfrecords (.tfrecords)
format as well as a numpy format (.npy).
TODO:
  * Implement dot dataset
  * Implement the egomotion dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import collections
import datetime
import glob
import numpy as np
import os
import csv
import scipy.io as sio
import scipy.misc as misc
import tensorflow as tf

import options
import utils

slim = tf.contrib.slim

# Helper functions
def csv_read(fname):
  csv_list = []
  with open(fname, 'rb') as f:
    csvreader = csv.reader(f)
    csv_list = list(csvreader)
  return csv_list

# Tensorflow features
def _bytes_feature(value):
  """Create arbitrary tensor Tensorflow feature."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Classes
# TODO: Make class for float and feature
class TensorFeature(slim.tfexample_decoder.ItemHandler):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super(TensorFeature, self).__init__(key)
    self._key = key
    self._shape = shape
    self._dtype = dtype
    self._description = description

  def get_feature_write(self, value):
    v = value.astype(self._dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self._dtype)
    return tf.reshape(tensor, self._shape)


class Dataset(object):
  """Class that manages creating, loading, and formatting the dataset."""
  # Constants
  MAX_IDX = 5000

  def __init__(self, opts):
    self.opts = opts
    self.dataset_dir = opts.dataset_dir
    self.type = opts.tf_type
    self.features = {
      'flow': TensorFeature(key='flow',
                            shape=opts.img_size + [4],
                            dtype=self.type,
                            description='Array of flow (u,v,x,y) values'),
      'foe': TensorFeature(key='foe',
                           shape=[3],
                           dtype=self.type,
                           description='Heading direction value'),
      'omega': TensorFeature(key='omega',
                             shape=[3],
                             dtype=self.type,
                             description='Angular velocity value'),
    }

  def process_features(self, loaded_features):
    features = {}
    for k, feat in self.features.iteritems():
      features[k] = feat.get_feature_write(loaded_features[k])
    return features

  def augment(self, keys, values):
    return keys, values

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
    matfiles = glob.glob(os.path.join(self.dataset_dir, mode, "[0-9]*.mat"))

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm(range(len(matfiles))):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = sio.loadmat(matfiles[index])
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()

    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    assert mode in self.opts.sample_sizes, "Mode {} not supported".format(mode)
    batch_size = self.opts.batch_size
    data_source_name = mode + '-[0-9][0-9].tfrecords'
    data_sources = glob.glob(os.path.join(self.dataset_dir, data_source_name))
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read()
                         for k, v in self.features.iteritems() }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      self.features)
    items_to_descriptions = { k: v._description
                              for k, v in self.features.iteritems() }
    dataset = slim.dataset.Dataset(
                data_sources=data_sources,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=self.opts.sample_sizes[mode],
                items_to_descriptions=items_to_descriptions)
    provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=self.opts.num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size,
                shuffle=self.opts.shuffle_data)
    # Extract features
    keys = self.features.keys()
    values = provider.get(keys)
    keys, values = self.augment(keys, values)
    # Flow preprocessing here?
    values = tf.train.batch(
                values,
                batch_size=batch_size,
                num_threads=self.opts.num_preprocessing_threads,
                capacity=5 * batch_size)
    return self.augment(dict(zip(keys, values)))

class CityscapesDataset(Dataset):
  def __init__(self, opts):
    super(CityscapesDataset, self).__init__(opts)
    self.nclasses = opts.nclasses
    self.convert_dir = opts.convert_dir
    self.img_size = opts.img_size
    self.net_img_size = opts.net_img_size
    self.crop_percent_variance = opts.crop_percent_variance
    self.features = {
      'image': TensorFeature(key='image',
                            shape=opts.img_size + [3],
                            dtype='uint8',
                            description='Array of (r,g,b) values'),
      'segmentation': TensorFeature(key='image',
                                    shape=opts.img_size,
                                    dtype='uint8',
                                    description='Array of segmentation labels'),
    }

  # TODO: Add color transforms
  def augment(self, keys, values):
    sample = dict(zip(keys,values))
    coin_flip = np.random.randn() > 0
    # var = self.crop_percent_variance
    # rand_h = (1 + (2*tf.random_uniform([1])-1)*var)
    # rand_w = (1 + (2*tf.random_uniform([1])-1)*var)
    # crop_h = tf.cast(self.img_size[0]*rand_h, 'int32')
    # crop_w = tf.cast(self.img_size[1]*rand_w, 'int32')
    # s_y = (self.img_size[0] - crop_h)
    # s_x = (self.img_size[1] - crop_w)
    # rand_y = tf.cast(s_y, 'float32')*tf.random_uniform([1])
    # rand_x = tf.cast(s_x, 'float32')*tf.random_uniform([1])
    # crop_y = tf.cast(rand_y, 'int32')
    # crop_x = tf.cast(rand_x, 'int32')

    # Prepare image and labels
    img = tf.image.convert_image_dtype(sample['image'], dtype=tf.float32)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    labels = tf.reshape(sample['segmentation'], self.img_size + [1])
    if coin_flip:
      img = tf.image.flip_left_right(img)
      labels = tf.image.flip_left_right(labels)
		img, labels = random_crop_and_pad_image_and_labels(img, labels, )
    print("GOT HERE")
    for v in [img, crop_y, crop_x, crop_h, crop_w]:
      print(v)
    # img = tf.image.crop_to_bounding_box(img, crop_y, crop_x, crop_h, crop_w)
    # img = tf.image.resize_images(img, [ self.net_img_size ])
    # labels = tf.image.crop_to_bounding_box(labels, crop_y, crop_x, crop_h, crop_w)
    # labels = tf.image.resize_images(labels, [ self.net_img_size ])
    sample['image'] = img
    sample['segmentation'] = labels

    return sample.keys(), sample.values()


  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
    csv_name = os.path.join(self.convert_dir, '{}_name_groups.csv'.format(mode))
    if not os.path.exists(csv_name):
      print("ERROR - no csv file with names")
    name_groups = csv_read(csv_name)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm(range(len(name_groups))):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      img_name = name_groups[index][0]
      seg_name = name_groups[index][2]
      loaded_features = {
        'image': misc.imread(os.path.join(self.convert_dir, img_name)),
        'segmentation': misc.imread(os.path.join(self.convert_dir, seg_name)),
      }
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()

    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_string = str(datetime.datetime.now())
      date_file.write('TFrecord created {}'.format(date_string))

def get_dataset(opts):
  return CityscapesDataset(opts)

if __name__ == '__main__':
  opts = options.get_opts()
  if opts.debug:
    SEQ_TYPES = ['train',]
  else:
    SEQ_TYPES = ['test', 'train']
  dataset = get_dataset(opts)
  for idx, seq_type in enumerate(SEQ_TYPES):
    dataset.convert_dataset(opts.dataset_dir, seq_type)


