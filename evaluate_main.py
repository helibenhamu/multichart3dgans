from utils.config import process_config
from utils.utils import get_args
import os
import scipy.io as sio


def main():
    # capture the config path from the run arguments
    args = get_args()
    # process the json configuration file
    config = process_config(args.config)

    # configure devices
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    import tensorflow as tf
    from data_loader.data_loader import DataLoader
    from models.gan_model import GANModel

    # set GPUS configuration
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpus
    gpuconfig.gpu_options.allow_growth = True

    # create tensorflow session
    sess = tf.Session(config=gpuconfig)
    # create your data generator
    data = DataLoader(config)

    # create an instance of the model
    model = GANModel(data, config)
    # load model
    model.load(sess)

    # generate random noise vector (could replace by a specified noise)
    noise = sess.run(model.noise)
    generated_charts = sess.run(model.generator(tf.convert_to_tensor(noise)))
    # create generations folder
    generations_path = os.path.join('experiments', config.exp_name, 'generations')
    os.mkdir(generations_path)
    sio.savemat(os.path.join(generations_path, 'generated_charts'), {'generated_charts': generated_charts})

    print('done')


if __name__ == '__main__':
    main()