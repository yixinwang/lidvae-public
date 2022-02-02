import os
import time
from datetime import datetime
import numpy as np
import numpy.random as npr
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from tensorflow.contrib.eager.python import tfe
from scipy.stats import mode
from scipy.io import loadmat
from scipy.special import expit

from tensorflow.python.keras.constraints import nonneg

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import argparse

# always reload dependency code
import importlib
import utils
import idgmmvae
import gmmvae
importlib.reload(utils)
importlib.reload(idgmmvae) 
importlib.reload(gmmvae) 
from utils import *
from idgmmvae import *
from gmmvae import *



config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.allow_soft_placement = True

# enable eager mode
tf.enable_eager_execution(config=config)


# set random seed
timenow = int(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
randseed = int(timenow % 10000)
print("Random seed", randseed)
tf.set_random_seed(randseed)
np.random.seed(randseed)


parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--discretez_dim', \
        type=int, default=10)
parser.add_argument('-hd', '--h_dim', \
        type=int, default=512)
parser.add_argument('-cd', '--contiz_dim', \
        type=int, default=64)
parser.add_argument('-od', '--out_dim', \
        type=int, default=784)
parser.add_argument('-opt', '--optimizer', \
        type=str, default="adam")
parser.add_argument('-lr', '--lr', \
        type=float, default=0.001)
parser.add_argument('-mmt', '--momentum', \
        type=float, default=0.5)
parser.add_argument('-ep', '--num_epochs', \
        type=int, default=1000)
parser.add_argument('-l2', '--l2reg', \
        type=float, default=0.001)
parser.add_argument('-nfd', '--nf_dim', \
        type=int, default=128)
parser.add_argument('-nb', '--num_bijectors', \
        type=int, default=2)
parser.add_argument('-gc', '--grad_clip', \
        type=float, default=5.)
parser.add_argument('-nl', '--gen_model_hidden_layers', \
        type=int, default=1)
parser.add_argument('-dt', '--data', \
        type=str, default="mnist", choices=['mnist', 'omniglot', 'fashionmnist', 'pinwheel'])
parser.add_argument('-md', '--model', \
        type=str, default='idvae', choices=['idvae', 'vae', 'laggingvae', 'betavae', 'li_et_al'])
parser.add_argument('--betavae', type=float, default=0.2, help='the beta value of betavae')
parser.add_argument('-ss', '--subsample', \
        type=int, default=20000)
# parameter for Li et al. 2019
parser.add_argument("--fb", type=int, default=0,
                 help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
parser.add_argument("--target_kl", type=float, default=7.,
                 help="target kl of the free bits trick, value 7 or 3 in Li et al.")

args, unknown = parser.parse_known_args()


discretez_dim = args.discretez_dim
h_dim = args.h_dim

momentum = args.momentum
lr = args.lr

num_epochs = args.num_epochs

batch_size = 100

nf_dim = args.nf_dim

l2reg = args.l2reg
regularizer = tf.keras.regularizers.l2(l2reg)

num_bijectors = args.num_bijectors
MODEL = 'RealNVP'
USE_BATCHNORM = False

grad_clip = args.grad_clip

qy_num_hidden_layers = 2
qz_num_hidden_layers = 2
gen_model_hidden_layers = args.gen_model_hidden_layers

dataname = args.data

train_data, train_labels, test_data, test_labels = load_data(dataname)

train_idx = npr.choice(train_data.shape[0], args.subsample, replace=False)
train_data = train_data[train_idx,:]
train_labels = train_labels[train_idx,:]

optimizer = set_optimizer(args.optimizer, lr, momentum)



contiz_dim = args.contiz_dim # the same as data dimension
out_dim = train_data.shape[1]

outfilename = 'out_'+dataname+'_'+args.model+str(timenow)

checkpoint_path = '../../res/'+dataname+'_gmm_' + args.model + 'lr' + str(args.lr) + 'l2reg' + str(args.l2reg) + 'seed' + str(timenow)+'/cp-{itr:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

# with tf.device(device):
# build model and optimizer
if args.model == 'idvae':
        kl_weight = 1.0
        model = idgmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers, gen_model_hidden_layers, USE_BATCHNORM)
elif args.model == 'vae':
        kl_weight = 1.0
        model = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers,  gen_model_hidden_layers, kl_weight=kl_weight, USE_BATCHNORM=USE_BATCHNORM)
elif args.model == 'betavae':
        kl_weight = args.betavae
        model = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers,  gen_model_hidden_layers, kl_weight=kl_weight, USE_BATCHNORM=USE_BATCHNORM)
elif args.model == 'li_et_al':
        kl_weight = 1.
        
        ae_model = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers,  gen_model_hidden_layers, kl_weight=0., USE_BATCHNORM=USE_BATCHNORM)
        
        vae_model = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers,  gen_model_hidden_layers, kl_weight=1., fb=2., target_kl=args.target_kl, USE_BATCHNORM=USE_BATCHNORM)
elif args.model == 'laggingvae':
        kl_weight = 1.
        vae_model1 = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers, gen_model_hidden_layers, kl_weight=0., USE_BATCHNORM=USE_BATCHNORM)
        vae_model2 = gmmvae(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers,  gen_model_hidden_layers, kl_weight=1., USE_BATCHNORM=USE_BATCHNORM)
for outfile in [outfilename+'.log', outfilename+'_au_discrete.log', outfilename+'_au_conti.log']:

    with open(outfile, 'a') as f:
        f.write("starting")    
        f.write("\n"+dataname+"_"+args.model+"\n")  
        f.write("\noptimizer")
        f.write(str(args))
        f.write("\nl2reg" + str(l2reg))
        f.write("\nkl_weight" + str(kl_weight))
        f.write("\nnf_dim" + str(nf_dim))
        f.write("\nnf_model" + str(MODEL))
        f.write("\nnum_bijectors" + str(num_bijectors))
        f.write("\nbatch_norm" + str(USE_BATCHNORM))
        f.write("\ngrad_clip" + str(grad_clip))
        f.write("\nrand_seed" + str(randseed))
        f.write("\ngen_model_hidden_layers" + str(gen_model_hidden_layers))
        f.write("\n\n\n")


if args.model == 'idvae':
        trained_model = \
                train_idgmmvae(model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path)
elif args.model == 'vae':
        trained_model = \
                train_gmmvae(model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path)
elif args.model == 'betavae':
        trained_model = \
                train_gmmvae(model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path)

elif args.model == 'li_et_al':
        print("initializing with autoencoder")
        trained_ae_model = \
                train_gmmvae(ae_model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, 100, outfilename, device, grad_clip, checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir) 
        vae_model.load_weights(latest)
        print("fitting variational autoencoder with KL truncation")
        trained_vae_model = \
                train_gmmvae(vae_model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path)
elif args.model == 'laggingvae':
        print("perform lagging inference (aggressivesly optimize inference network")
        trained_vae_model1 = \
                train_gmmvae(vae_model1, train_data, train_labels, test_data, test_labels, optimizer, batch_size, 100, outfilename, device, grad_clip, checkpoint_path, lagging=1)

        latest = tf.train.latest_checkpoint(checkpoint_dir) 
        print("perform standard inference")
        vae_model2.load_weights(latest)
        trained_vae_model2 = \
                train_gmmvae(vae_model2, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path, lagging=0)
                          



