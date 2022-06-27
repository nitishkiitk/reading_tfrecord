#Import the libraries required
import tensorflow as tf

import numpy as np
import IPython.display as display

#load your file
filenames = "test_2.tfrecord"
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)