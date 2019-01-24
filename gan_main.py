import os
from utils.config import process_config
from utils.logger import Logger
from utils.utils import get_args
from shutil import copyfile


def main():
    # capture the config path from the run arguments
    # then process the json configuration file

    args = get_args()

    config = process_config(args.config)
    try:
        copyfile(args.config, config.results_dir + '/config.json')
    except:
        print("error updating config file")
        # TODO: handle config file copying

    # configure devices
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    # import objects
    from data_loader.data_loader import DataLoader
    from models.gan_model import GANModel
    from trainers.gan_trainer import GANTrainer
    import tensorflow as tf

    # set GPUS configuration
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpus
    gpuconfig.gpu_options.allow_growth = True

    # set random seed for tensorflow
    tf.set_random_seed(config.seed)

    # create tensorflow session
    sess = tf.Session(config=gpuconfig)
    # create data loader
    data = DataLoader(config)
    # create an instance of the model
    model = GANModel(data, config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)

    # train the model
    trainer.train()


if __name__ == '__main__':
    main()
