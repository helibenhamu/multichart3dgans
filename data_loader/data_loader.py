# DataLoader class - handles data loading
#       creates dataset object within tensorflow dataset API

import tensorflow as tf
import multiprocessing


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.set_dataset()

    def set_dataset(self):
        '''
        creates a tensorflow dataset object
        initializable iterator with placeholder input for filenames
        '''

        self.filenames = tf.placeholder(tf.string, shape=[None])   # filenames placeholder
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_function, num_parallel_calls=multiprocessing.cpu_count())  # Parse the record into tensors.
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.config.dataLoader_buffer)
        dataset = dataset.batch(self.config.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.get_next = tf.decode_raw(self.iterator.get_next(), tf.float32)


    @staticmethod
    def _parse_function(example_proto):
        features = {"image_string": tf.FixedLenFeature((), tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["image_string"]



