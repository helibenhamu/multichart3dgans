# Download human models data and a pre-trained model

import os

# download human models data and place at:
# databases/tfrecords/humans_64x64
www = 'https://www.dropbox.com/s/0xznbd147tp624c/humans_64x64.tar.gz'
tarfile = os.path.basename(www)

os.system('wget --no-check-certificate %s; tar --extract --file %s' % (www, tarfile))

tfrecords_dir = 'databases/tfrecords'
os.mkdir(tfrecords_dir)
os.system('mv %s %s' % (tarfile[:-7], tfrecords_dir))
os.system('rm %s' % (tarfile))

# download a pre-trained model on the above data
# place at:
# experiments/humans_gan_pretrained
www = 'https://www.dropbox.com/s/dxqo7uufkv7hmxt/humans_gan_pretrained.tar.gz'
tarfile = os.path.basename(www)
os.system('wget --no-check-certificate %s; tar --extract --file %s' % (www, tarfile))

experiments_dir = 'experiments'
os.system('mv %s %s' % (tarfile[:-7], experiments_dir))
os.system('rm %s' % (tarfile))
