import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import random
import glob
import itertools
import operator


def main(src_dir, dst_dir, file_name_prefix='data', print_period=1000, tfr_size=500, train_size_percent=95):
    """ main:
        gather all the images from src_dir and save them into tfrecords in dst_dir
        the maximum num of examples in each tfr will be tfr_size.

        Arguments:
            1. src_dir: the directory with the flattened images in .mat format
            2. dst_dir: the tfrecords directory to save the tfrecords in

        Process:
        1. Gather all the .mat files
        2. split them into val and train (train_size, num_files-train_size)
        3. turn mat files into ndarrays
        4. concatenate the ndarrays in batches of tfr_size in the shape [tfr_size, image_numel]
        5. save each batch into a tfrrecord
    """
    print('converting {0} to {1}'.format(src_dir, dst_dir))
    # 1. Gather all the .mat files
    files = glob.glob(os.path.join(src_dir, '**/*.mat'), recursive=True)
    print(len(files))

    train_size = int((train_size_percent*len(files))/100)

    # 2. Split the files into val and train
    random.shuffle(files)
    train_files = files[: train_size]
    val_files = files[train_size:]
    all_files = [val_files, train_files]

    # convert each dataset into tfrecords
    dst_dir_val = os.path.join(dst_dir, 'val')
    dst_dir_train = os.path.join(dst_dir, 'train')
    dst_dirs = [dst_dir_val, dst_dir_train]

    # save info file
    dictionary = {'train_size': train_size,
                  'val_size': len(files)-train_size,
                  'dataset_size': len(files)}

    create_or_recreate_dir(dst_dir)
    np.save(os.path.join(dst_dir, 'info.npy'), dictionary)

    for files, dst_dir, data_type in zip(all_files, dst_dirs, ['val', 'train']):
        print('converting {0} dataset of size {1}'.format(data_type, len(files)))
        create_or_recreate_dir(dst_dir)
        turn_dataset_to_tfrecords(files=files, dst_dir=dst_dir, tfr_size=tfr_size, file_name_prefix=file_name_prefix, print_period=print_period)


def create_or_recreate_dir(dir):
    import shutil
    if os.path.isdir(dir):
        #shutil.rmtree(dir)
        return
    os.makedirs(dir)


def turn_dataset_to_tfrecords(files, dst_dir, tfr_size, file_name_prefix, print_period):
    tfr_counter = 0
    data_list = []
    file_paths_list = []
    for index, file_path in enumerate(files):
        # read mat file into ndarray
        print(file_path)
        mat_content = sio.loadmat(file_path)
        data = mat_content['data']
        # append to images list
        data_list.append(data)
        # append path to paths list
        file_paths_list.append(file_path)

        if (index % print_period == 0) and index > 0:
            print('gathered {0} images to turn to tfrecord'.format(index))

        if index % tfr_size == 0 and index > 0:
            tfr_path = os.path.join(dst_dir, '{0}_{1}.tfrecord'.format(file_name_prefix, tfr_counter))
            create_tfr(data_list, file_paths_list, tfr_path)
            # increase the tfr counter
            tfr_counter += 1
            # restart the images list
            data_list = []

    # if the list has remaining images (less than tfr_size)
    # save it into another tfr
    if len(data_list) != 0:
        tfr_path = os.path.join(dst_dir, '{0}_{1}.tfrecord'.format(file_name_prefix, tfr_counter))
        create_tfr(data_list, file_paths_list, tfr_path)


def create_tfr(data_list, file_paths_list, tfr_path):
    writer = tf.python_io.TFRecordWriter(tfr_path)

    for image_np, path in zip(data_list, file_paths_list):
        # conver the image to string
        image_string = image_np.tostring()
        # make the example from the image and it's path
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'image_string': _bytes_feature(image_string)}
            ))
        # write the example to the tfr
        writer.write(example.SerializeToString())

    # close the tfr
    writer.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':

    database_signature = 'humans_64x64'
    src_dir = 'databases/images/' + database_signature
    dst_dir_base = 'databases/tfrecords/' + database_signature
    file_name_prefix = 'data'

    main(src_dir=src_dir, dst_dir=dst_dir_base, file_name_prefix=file_name_prefix, train_size_percent=100)


