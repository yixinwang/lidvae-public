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

from tensorflow.python.keras.constraints import nonneg

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import argparse

# always reload dependency code
import importlib
import utils
importlib.reload(utils)
from utils import *

class idgmmvae(tf.keras.Model):
    def __init__(self, discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers, px_num_hidden_layers, USE_BATCHNORM=False):
        super(idgmmvae, self).__init__(discretez_dim, contiz_dim, h_dim, out_dim, regularizer, num_bijectors, nf_dim, qy_num_hidden_layers, qz_num_hidden_layers, px_num_hidden_layers, USE_BATCHNORM)
        self.discretez_dim = discretez_dim
        self.contiz_dim = contiz_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.qy_num_hidden_layers = qy_num_hidden_layers

        self.qy_layers = []
        for i in range(self.qy_num_hidden_layers):
            self.qy_layers.append(tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer))

        self.qz_num_hidden_layers = qz_num_hidden_layers

        self.qz_layers = []
        for i in range(self.qz_num_hidden_layers):
            self.qz_layers.append(tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer))


        '''
        input convex neural network
        input: y
        parameters: theta = {W^y_{0:k-1}, W^z_{1:k-1}, b_{0:k-1}}
        initial condition: z_0 = 0, W_0^z = 0
        intermediate layer: z_{i+1} = g_i(W_i^z z_i + W_i^y y + b_i), 
                            i=0, ..., k-1
        final layer: f(y,theta) = z_k
        constraints: W^z_{1:k-1} is nonnegative
                     g_i convex and non-decreasing
        '''
        # Wy_i involves W_i^y, b_i
        self.px1_num_hidden_layers = px_num_hidden_layers
        self.px2_num_hidden_layers = px_num_hidden_layers
        # self.px3_num_hidden_layers = px_num_hidden_layers

        # the first set of icnn
        # first layer is treated separately
        self.icnn1_Wy0 = tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer)

        self.icnn1_Wy_layers = []
        self.icnn1_Wz_layers = []
        for i in range(self.px1_num_hidden_layers-1):
            self.icnn1_Wy_layers.append(tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer))
            self.icnn1_Wz_layers.append(tf.keras.layers.Dense(h_dim, 
            use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))

        # add final layer with output dimension = 1
        self.icnn1_Wy_layers.append(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))
        self.icnn1_Wz_layers.append(tf.keras.layers.Dense(1, 
            use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))



        # the second set of icnn
        # first layer is treated separately
        # self.icnn2_Wy0 = tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer)
        self.icnn2_Wy0 = tf.keras.layers.Dense(out_dim, kernel_regularizer=regularizer)

        self.icnn2_Wy_layers = []
        self.icnn2_Wz_layers = []
        for i in range(self.px2_num_hidden_layers-1):
            self.icnn2_Wy_layers.append(tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer))
            self.icnn2_Wz_layers.append(tf.keras.layers.Dense(h_dim, 
            use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))

        # add final layer with output dimension = 1
        self.icnn2_Wy_layers.append(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))
        self.icnn2_Wz_layers.append(tf.keras.layers.Dense(1, 
            use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))


        # the third set of icnn
        # first layer is treated separately
        # self.icnn3_Wy0 = tf.keras.layers.Dense(out_dim, kernel_regularizer=regularizer)

        # self.icnn3_Wy_layers = []
        # self.icnn3_Wz_layers = []
        # for i in range(self.px3_num_hidden_layers-1):
        #     self.icnn3_Wy_layers.append(tf.keras.layers.Dense(h_dim, kernel_regularizer=regularizer))
        #     self.icnn3_Wz_layers.append(tf.keras.layers.Dense(h_dim, 
        #     use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))

        # # add final layer with output dimension = 1
        # self.icnn3_Wy_layers.append(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))
        # self.icnn3_Wz_layers.append(tf.keras.layers.Dense(1, 
        #     use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), kernel_regularizer=regularizer))


        self.fc2 = tf.keras.layers.Dense(discretez_dim, kernel_regularizer=regularizer)
        self.fc5 = tf.keras.layers.Dense(contiz_dim, kernel_regularizer=regularizer)
        self.fc6 = tf.keras.layers.Dense(contiz_dim, kernel_regularizer=regularizer)
        
        self.fc7 = tf.keras.layers.Dense(contiz_dim, kernel_regularizer=regularizer)
        self.fc8 = tf.keras.layers.Dense(contiz_dim, kernel_regularizer=regularizer)

        # self.fc9 = tf.keras.layers.Dense(h_dim, 
            # kernel_regularizer=regularizer)
        self.fc9 = tf.keras.layers.Dense(self.out_dim, 
            kernel_regularizer=regularizer)
        
        # self.fc10 = tf.keras.layers.Dense(self.out_dim, kernel_regularizer=regularizer)        

        self.shift_and_log_scale_fn = []
        for i in range(num_bijectors):
            self.shift_and_log_scale_fn.append(tfb.real_nvp_default_template(
                                hidden_layers=[nf_dim, nf_dim], shift_only=True)) 
        bijectors = []
        for i in range(num_bijectors):
            bijectors.append(tfb.RealNVP(shift_and_log_scale_fn=self.shift_and_log_scale_fn[i], num_masked=2))
            if USE_BATCHNORM and i % 2 == 0:
                # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
                bijectors.append(tfb.BatchNormalization())
        self.bijector = tfb.Chain(list(reversed(bijectors)))

    def icnn1_grad(self, x_train_tensor):
        with tf.GradientTape() as icnn1_tape:
            icnn1_tape.watch(x_train_tensor)
            h = [[None] for i in range(self.px1_num_hidden_layers + 1)]
            h[0] = tf.square(tf.nn.leaky_relu(self.icnn1_Wy0(x_train_tensor)))
            for i in range(self.px1_num_hidden_layers):
                h[i+1] = tf.nn.leaky_relu(self.icnn1_Wz_layers[i](h[i]) + self.icnn1_Wy_layers[i](x_train_tensor))
        dout_dx = icnn1_tape.gradient(h[-1], x_train_tensor)
        return dout_dx


    def icnn2_grad(self, x_train_tensor):
        with tf.GradientTape() as icnn2_tape:
            icnn2_tape.watch(x_train_tensor)
            h = [[None] for i in range(self.px2_num_hidden_layers + 1)]
            h[0] = tf.square(tf.nn.leaky_relu(self.icnn2_Wy0(x_train_tensor)))
            for i in range(self.px2_num_hidden_layers):
                h[i+1] = tf.nn.leaky_relu(self.icnn2_Wz_layers[i](h[i]) + self.icnn2_Wy_layers[i](x_train_tensor))
        dout_dx = icnn2_tape.gradient(h[-1], x_train_tensor)
        return dout_dx

    def icnn3_grad(self, x_train_tensor):
        with tf.GradientTape() as icnn3_tape:
            icnn3_tape.watch(x_train_tensor)
            h = [[None] for i in range(self.px3_num_hidden_layers + 1)]
            h[0] = tf.square(tf.nn.leaky_relu(self.icnn3_Wy0(x_train_tensor)))
            for i in range(self.px3_num_hidden_layers):
                h[i+1] = tf.nn.leaky_relu(self.icnn3_Wz_layers[i](h[i]) + self.icnn3_Wy_layers[i](x_train_tensor))
        dout_dx = icnn3_tape.gradient(h[-1], x_train_tensor)
        return dout_dx


    def qy_graph(self, x, k):
        h = [[None] for i in range(self.qy_num_hidden_layers + 1)]
        h[0] = x
        for i in range(self.qy_num_hidden_layers):
            h[i+1] = tf.nn.relu(self.qy_layers[i](h[i]))
        qy_logit = self.fc2(h[-1])
        qy = tf.nn.softmax(qy_logit)
        return qy_logit, qy

    def nfdist(self, zm, zv):
        # normalizing flow variational distribution for conti_z
        nfdist = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=zm, scale_diag=zv),
            bijector=self.bijector)
        return nfdist         

    def qz_graph(self, x, y):
        xy = tf.concat([x, y], 1)
        h = [[None] for i in range(self.qz_num_hidden_layers + 1)]
        h[0] = xy
        for i in range(self.qz_num_hidden_layers):
            h[i+1] = tf.nn.relu(self.qz_layers[i](h[i]))
        zm = self.fc5(h[-1])
        zv = tf.nn.softplus(self.fc6(h[-1]))
        nfdist = self.nfdist(zm, zv)
        z = nfdist.sample()
        return z, zm, zv
    
    def px_graph(self, z, y):
        zm = self.fc7(y)
        # zm = tf.pad(y * 10 + 1, [[0,0], [0, self.contiz_dim - self.discretez_dim]], "CONSTANT")
        # print(y * 100, self.contiz_dim, self.discretez_dim)
        # print(zm) 
        zv = tf.nn.softplus(self.fc8(y))



        # need to do above to create the layers first

        # zm = tf.matmul(y, tf.cumsum(tf.nn.softplus(self.fc7.kernel), axis=0))
        
        # zv = tf.nn.softplus(tf.matmul(y, tf.cumsum(tf.nn.softplus(self.fc7.kernel), axis=0)))

        # print("z", z)
        z_mid1 = self.icnn1_grad(z) # note changes in here should change loss gradient below too
        # print("z_mid1", z_mid1)


        z_mid2 = self.fc9(z_mid1)

        # print(self.fc9.weights)

        # make sure the weights are full rank
        
        s, u, v = tf.linalg.svd(self.fc9.weights[0])

        s_fr = tf.clip_by_value(s, clip_value_min=1e-5, clip_value_max=10.)

        self.fc9.weights[0] = tf.matmul(u, tf.matmul(tf.linalg.diag(s_fr), v, adjoint_b=True))

        # print(s, s_fr)
        # print(self.fc9.weights)

        # beta_zmid1_zmid2 = tf.eye(self.contiz_dim, self.h_dim)
        # z_mid2 = tf.matmul(z_mid1, beta_zmid1_zmid2)

        # print("z_mid2", z_mid2)
        z_mid3 = self.icnn2_grad(z_mid2)
        # px_logit = z_mid3

        # z_mid4 = self.fc10(z_mid3)

        # beta_zmid2_zmid3 = tf.eye(self.h_dim, self.out_dim)
        # z_mid4 = tf.matmul(z_mid3, beta_zmid2_zmid3)
        # z_mid5 = self.icnn3_grad(z_mid4)
        # px_logit = z_mid5
        px_logit = z_mid3

        # z_mid4 = self.fc10(z_mid3)
        # z_mid5 = self.icnn3_grad(z_mid4)
        # px_logit = z_mid5
        return zm, zv, px_logit

    def labeled_loss(self, x, px_logit, z, zm, zv, zm_prior, zv_prior):
        xy_loss = -log_bernoulli_with_logits(x, px_logit)
        xy_loss += -tfd.MultivariateNormalDiag(
            loc=zm_prior, scale_diag=zv_prior).log_prob(z)
        nfdist = self.nfdist(zm, zv)
        xy_loss += nfdist.log_prob(z)
        xy_loss += -np.log(1./self.discretez_dim)
        return xy_loss
        

    def call(self, x):
        xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
        qy_logit, qy = self.qy_graph(xb, k=self.discretez_dim)
        
        z, zm, zv, zm_prior, zv_prior, px_logit = [[None] * self.discretez_dim for i in range(6)]
        y_ = tf.fill(tf.stack([tf.shape(x)[0], self.discretez_dim]), 0.0)
        for i in range(self.discretez_dim):
            y = tf.add(y_, tf.constant(np.eye(self.discretez_dim)[i], dtype='float32'))
            z[i], zm[i], zv[i] = self.qz_graph(xb, y)
            zm_prior[i], zv_prior[i], px_logit[i] = self.px_graph(z[i], y)
        return xb, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, px_logit


def iw_nll(model, images, iw_nsamples=100):
    loglikeratios_list = []
    for i in range(iw_nsamples):
        xb, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, px_logit = model(images)    
        losses = [None] * model.discretez_dim
        for i in range(model.discretez_dim):
            losses[i] = model.labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
        loss = tf.reshape(-tf.reduce_mean(tf.add_n([qy[:, i] * losses[i] for i in range(model.discretez_dim)])), [1])      
        # print("loss", loss)
        loglikeratios_list.append(loss)
    # print("list", loglikeratios_list)
    loglikeratios = tf.concat(loglikeratios_list, 0)
    iwae_nll = tf.math.reduce_logsumexp(loglikeratios,0) - tf.math.log(tf.constant([iw_nsamples],dtype=tf.float32))
    return iwae_nll

def nent_and_loss(model, images):
    xb, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, px_logit = model(images)
    nent = -cross_entropy_with_logits(qy_logit, qy)
    losses = [None] * model.discretez_dim
    for i in range(model.discretez_dim):
        losses[i] = model.labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
    loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(model.discretez_dim)])

    kl_discrete = nent - np.log(1./model.discretez_dim)

    kl_contis = [[None] for i in range(model.discretez_dim)]

    for i in range(model.discretez_dim):
        kl_contis[i] = model.nfdist(zm[i], zv[i]).log_prob(z) - \
            tfd.MultivariateNormalDiag(
            loc=zm_prior[i], scale_diag=zv_prior[i]).log_prob(z)
            
    kl_conti = tf.add_n([qy[:, i] * kl_contis[i] for i in range(model.discretez_dim)])

    au_discrete = tf.math.reduce_std(qy, 0)

    au_conti = tf.math.reduce_std(tf.add_n(
        [tf.multiply(tf.expand_dims(qy[:, i], 1), zm[i]) for i in range(model.discretez_dim)]), 0)

    return nent, loss, kl_discrete, au_discrete, kl_conti, au_conti, qy_logit


def eval_model(model, train_data, train_labels, test_data, test_labels, itr, outfilename):

    with tf.device('/cpu:0'):
        train_evalset = np.random.choice(train_data.shape[0], 1000)
        test_evalset = np.random.choice(test_data.shape[0], 1000)
        train_images = train_data[train_evalset]
        train_labels = train_labels.argmax(1)[train_evalset]
        test_images = test_data[test_evalset]
        test_labels = test_labels.argmax(1)[test_evalset]

        train_nent, train_loss, train_kl_discrete, train_au_discrete, train_kl_conti, train_au_conti, train_qy_logit = nent_and_loss(model, train_images)
        test_nent, test_loss, test_kl_discrete, test_au_discrete, test_kl_conti, test_au_conti, test_qy_logit = nent_and_loss(model, test_images)

        train_iwnll = iw_nll(model, train_images).numpy()[0]
        test_iwnll = iw_nll(model, test_images).numpy()[0]

        train_ent, train_loss, train_kl_discrete, train_au_discrete, train_kl_conti, train_au_conti = -train_nent.numpy().mean(), train_loss.numpy().mean(), train_kl_discrete.numpy().mean(), train_au_discrete.numpy(), train_kl_conti.numpy().mean(), train_au_conti.numpy()

        test_ent, test_loss, test_kl_discrete, test_au_discrete, test_kl_conti, test_au_conti = -test_nent.numpy().mean(), test_loss.numpy().mean(), test_kl_discrete.numpy().mean(), test_au_discrete.numpy(), test_kl_conti.numpy().mean(), test_au_conti.numpy()
        
        zacc_train, zacc_test = -1, -1
        if test_labels is not None:
            if train_labels is not None:
                zacc_test = z_testacc(test_qy_logit, test_labels)
                zacc_train = z_testacc(train_qy_logit, train_labels)


        # print(itr)
        with open(outfilename+'.log', 'a') as f:
            f.write("\n\nItr" + str([itr]) + 
                "\ntrain ent loss zacc kl_discrete kl_conti iw_nll" + str([train_ent]+ [train_loss] + [zacc_train] + 
                    [train_kl_discrete] + [train_kl_conti] + [train_iwnll]) + 
                "\ntest ent loss zacc kl_discrete kl_conti iw_nll" + str([test_ent]+ [test_loss] + [zacc_test] + 
                    [test_kl_discrete] + [test_kl_conti] + [test_iwnll]))

        with open(outfilename+'_au_discrete.log', 'a') as f:
            f.write("\n\nItr" + str([itr]) + 
                "\ntrain au_discrete" + str([train_au_discrete]) + 
                "\ntest au_discrete" + str([test_au_discrete]))

        with open(outfilename+'_au_conti.log', 'a') as f:
            f.write("\n\nItr" + str([itr]) + 
                "\ntrain au_conti" + str([train_au_conti]) + 
                "\ntest au_conti" + str([test_au_conti]))

        print("\n\nItr", itr, 
            "\ntrain ent loss zacc kl_discrete kl_conti iw_nll", train_ent, train_loss, zacc_train, train_kl_discrete, train_kl_conti, train_iwnll,
            "\ntest ent loss zacc kl_discrete kl_conti iw_nll", test_ent, test_loss, zacc_test, test_kl_discrete, test_kl_conti, test_iwnll)

    return train_loss


def train_idgmmvae(model, train_data, train_labels, test_data, test_labels, optimizer, batch_size, num_epochs, outfilename, device, grad_clip, checkpoint_path):
    dataset = tf.data.Dataset.from_tensor_slices((train_data,))
    dataset = dataset.shuffle(batch_size * 5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    num_batches = train_data.shape[0] // batch_size


    for outfile in [outfilename+'.log', outfilename+'_au_discrete.log', outfilename+'_au_conti.log']:

        with open(outfile, 'a') as f:
            f.write("hi i'm starting")   
            f.write("idgmmvae_model") 
            f.write("\noptimizer")
            f.write("discretez_dim"+str(model.discretez_dim)+\
                "h_dim"+str(model.h_dim)+\
                "contiz_dim"+str(model.contiz_dim))
        

    for epoch in range(num_epochs):
        for batch, (images,) in enumerate(dataset):
            itr = epoch * num_batches + batch
            with tf.device(device):

                with tf.GradientTape() as loss_tape:
                    loss_tape.watch(model.variables)

                    xb, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, _ = model(images)
                    px_logit = [None] * model.discretez_dim
                    # we repeat the icnn calculation here. otherwise the
                    # nested gradient is not calculated correctly
                    for zi in range(model.discretez_dim):
                        # print("start", zi)
                        h1 = [[None] for i in range(model.px1_num_hidden_layers + 1)]
                        h2 = [[None] for i in range(model.px2_num_hidden_layers + 1)]
                        # h3 = [[None] for i in range(model.px3_num_hidden_layers + 1)]
                        hz = z[zi]
                        # print("hz", hz)
                        with tf.GradientTape() as loss_icnn1_tape:
                            loss_icnn1_tape.watch(hz)
                            h1[0] = tf.square(tf.nn.leaky_relu(model.icnn1_Wy0(hz)))
                            for i in range(model.px1_num_hidden_layers):
                                h1[i+1] = tf.nn.leaky_relu(model.icnn1_Wz_layers[i](h1[i]) + model.icnn1_Wy_layers[i](hz))
                        # print("h1[-1]", h1[-1])
                        h_mid1 = loss_icnn1_tape.gradient(h1[-1], hz)
                        # print("h_mid1", h_mid1)

                        h_mid2 = model.fc9(h_mid1)
                        
                        # beta_zmid1_zmid2 = tf.eye(model.contiz_dim, model.h_dim)
                        # h_mid2 = tf.matmul(h_mid1, beta_zmid1_zmid2)

                        # print("h_mid2", h_mid2)
                        with tf.GradientTape() as loss_icnn2_tape:
                            loss_icnn2_tape.watch(h_mid2)
                            h2[0] = tf.square(tf.nn.leaky_relu(model.icnn2_Wy0(h_mid2)))
                            for i in range(model.px2_num_hidden_layers):
                                h2[i+1] = tf.nn.leaky_relu(model.icnn2_Wz_layers[i](h2[i]) + model.icnn2_Wy_layers[i](h_mid2))
                        # print("h2[-1]", h2[-1])
                        h_mid3 = loss_icnn2_tape.gradient(h2[-1], h_mid2)

                        # beta_zmid2_zmid3 = tf.eye(model.h_dim, model.out_dim)
                        # h_mid4 = tf.matmul(h_mid3, beta_zmid2_zmid3)

                        # h_mid4 = model.fc10(h_mid3)
                        
                        # with tf.GradientTape() as loss_icnn3_tape:
                        #     loss_icnn3_tape.watch(h_mid4)
                        #     h3[0] = tf.square(tf.nn.leaky_relu(model.icnn3_Wy0(h_mid4)))
                        #     for i in range(model.px3_num_hidden_layers):
                        #         h3[i+1] = tf.nn.leaky_relu(model.icnn3_Wz_layers[i](h3[i]) + model.icnn3_Wy_layers[i](h_mid4))
                        # h_mid5 = loss_icnn3_tape.gradient(h3[-1], h_mid4)

                        # # print("h4", h4)
                        # px_logit[i] = h_mid5
                        # print(zi, h_mid3)
                        px_logit[zi] = h_mid3
                        # print(px_logit)

                    # Forward pass
                    
                    nent = -cross_entropy_with_logits(qy_logit, qy)
                    losses = [None] * model.discretez_dim
                    for i in range(model.discretez_dim):
                        losses[i] = model.labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                    loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(model.discretez_dim)])

                    # print(model.fc7.weights)
                    # S1 = tf.linalg.svd(model.fc7.weights[0], compute_uv=False)
                    # print(S1)
                    # S2 = tf.linalg.svd(model.fc9.weights[0], compute_uv=False)
                    # rank_penalty = tf.reduce_sum(tf.math.minimum(tf.math.log(S1), 1e-6 * tf.ones_like(S1))) + tf.reduce_sum(tf.math.minimum(tf.math.log(S2), 1e-6 * tf.ones_like(S2)))
                    # print(rank_penalty)
                    # loss -= rank_penalty

                gradients = loss_tape.gradient(loss, model.variables)

                capped_gradients = tf.clip_by_global_norm(gradients, grad_clip)[0]
                # gradient clipping is essential for normalizing flow

                grad_vars = zip(capped_gradients, model.variables)
                optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

            if itr %  num_batches == 0:
            # if itr %  10 == 0:

                train_loss = eval_model(model, train_data, train_labels, test_data, test_labels, itr, outfilename)
                model.save_weights(checkpoint_path.format(itr=itr))     

                if math.isnan(train_loss):
                    break

                # cmap=plt.cm.jet

                # if model.out_dim == 784:                    
                #     y_ = tf.fill(tf.stack([batch_size, model.discretez_dim]), 0.0)
                #     for i in range(model.discretez_dim):
                #         y = tf.add(y_, tf.constant(np.eye(model.discretez_dim)[i], dtype='float32'))
                #         zm_prior = model.fc7(y)
                #         zv_prior = tf.nn.softplus(model.fc8(y))
                #         z = tfd.MultivariateNormalDiag(loc=zm_prior, scale_diag=zv_prior).sample()
                #         _, _, px_logit = model.px_graph(z, y) 
                #         out = tf.math.reduce_mean(tf.nn.sigmoid(px_logit), 0)
                #         out = tf.reshape(out, [28, 28]).numpy() * 255
                #         out = out.astype(np.uint8)
                #         plt.imsave(os.path.dirname(checkpoint_path)+'/itr'+str(itr)+'_discretez'+str(i)+'.png', out, cmap=cmap)

    return model
