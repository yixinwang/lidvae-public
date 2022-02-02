# import torch

import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

from .decoder import DecoderBase
from .decoder_helper import BeamSearchNode

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_rv = nn.Parameter(torch.Tensor(out_features, in_features))
        # if use_bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_rv, 0.01)
        # nn.init.xavier_normal_(self.log_weight)

    def forward(self, input):
        # return nn.functional.linear(input, torch.clamp(self.log_weight.exp(), min=0.0, max=1e10), bias=self.bias)
        return nn.functional.linear(input, torch.clamp(self.weight_rv, min=0.01))


class FullRankLinear(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(FullRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).cuda()
        # self.alpha = 0.01
        # self.U_fullmat = nn.Parameter(torch.Tensor(out_features, in_features)).cuda()
        # self.V_fullmat = nn.Parameter(torch.Tensor(out_features, in_features)).cuda()
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None


        # self.reset_parameters()
        # self.U, self.diag, _ = torch.svd(self.U_fullmat + self.alpha*torch.eye(n=self.weight.shape[0], m=self.weight.shape[1]).cuda())
        # _, self.diag, self.V = torch.svd(self.V_fullmat + self.alpha*torch.eye(n=self.weight.shape[0], m=self.weight.shape[1]).cuda())

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, 0.01)
        # nn.init.xavier_uniform_(self.U_fullmat, 0.1)
        # nn.init.xavier_uniform_(self.V_fullmat, 0.1)
        # nn.init.xavier_normal_(self.log_weight)

    def forward(self, input, alpha=1.):
        # return nn.functional.linear(input, torch.clamp(self.log_weight.exp(), min=0.0, max=1e10), bias=self.bias)

        # take the upper triangular part, and then set diagonal=1
        # return nn.functional.linear(input, torch.triu(self.weight,diagonal=1) + alpha * torch.eye(n=self.weight.shape[0], m=self.weight.shape[1]).cuda())

        # return nn.functional.linear(input, torch.eye(n=self.weight.shape[0], m=self.weight.shape[1])%de)

        # self.weight[torch.isnan(self.weight)] = 1e-5
        
        # self.weight[self.weight!=self.weight] = 1e-5 # replace nan with 1e-5
        U, S, V = torch.svd(self.weight+1e-5*torch.eye(n=self.weight.shape[0], m=self.weight.shape[1]).cuda())
        # S_fr = torch.clamp(torch.abs(S), min=1e-5, max=10.) * torch.sign(S)
        S_fr = torch.clamp(S, min=1e-5, max=10.)
        # print(S, S_fr)

        # print(S, S_fr)
        # weight = torch.matmul(torch.matmul(self.U, torch.eye(n=self.diag.shape[0]).cuda()), self.V.T)
        # print(torch.dist(self.weight, U @ torch.diag_embed(S) @ V.T))
        return nn.functional.linear(input, U @ torch.diag_embed(S_fr) @ V.T, bias=self.bias)

class idLSTMDecoder(DecoderBase):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(idLSTMDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

        # concatenate z with input
        # self.lstm = nn.LSTM(input_size=args.ni + args.nz,
        #                     hidden_size=args.dec_nh,
        #                     num_layers=1,
        #                     batch_first=True)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.dec_nh,
                            num_layers=1,
                            batch_first=True)

        # prediction layer involves ICNN

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

        self.icnn1_num_layers = args.icnn_num_layers
        self.icnn2_num_layers = args.icnn_num_layers
        # self.icnn3_num_layers = args.icnn_num_layers

        # self.icnn1_inputdim = len(vocab)
        self.icnn1_inputdim = (args.dec_nh + args.nz)
        # self.icnn2_inputdim = args.icnn_nh
        self.icnn2_inputdim = len(vocab)
        
        # self.icnn3_inputdim = len(vocab)
        # self.icnn2_inputdim = (args.dec_nh + args.nz)

        self.icnn1_nh = args.icnn_nh # size of hidden layers
        self.icnn2_nh = args.icnn_nh
        # self.icnn3_nh = args.icnn_nh

        device = torch.device("cuda:0")

        # self.icnn1_icnn2 = UpperTriangleLinear(self.icnn1_inputdim, self.icnn2_inputdim).to(device) # use bias? linear transformation between two icnns
        # self.icnn2_icnn3 = UpperTriangleLinear(self.icnn2_inputdim, self.icnn3_inputdim).to(device)
        self.icnn1_icnn2 = FullRankLinear(self.icnn1_inputdim, self.icnn2_inputdim).to(device) # use bias? linear transformation between two icnns
        # self.icnn2_icnn3 = FullRankLinear(self.icnn2_inputdim, self.icnn3_inputdim).to(device)



        #######################3
        # the first set of icnn
        ########################

        # first layer is treated separately
        # self.icnn1_Wy0 = nn.Linear(self.icnn1_inputdim, self.icnn1_nh).to(device)
        self.icnn1_Wy0 = nn.Linear(self.icnn1_inputdim, self.icnn1_nh).to(device)

        self.icnn1_Wy_layers = []
        self.icnn1_Wz_layers = []
        for i in range(self.icnn1_num_layers-1):
            self.icnn1_Wy_layers.append(nn.Linear(self.icnn1_inputdim, self.icnn1_nh).to(device))
            self.icnn1_Wz_layers.append(PositiveLinear(self.icnn1_nh, self.icnn1_nh).to(device))

        # add final layer with output dimension = 1
        self.icnn1_Wy_layers.append(nn.Linear(self.icnn1_inputdim, 1).to(device))
        self.icnn1_Wz_layers.append(PositiveLinear(self.icnn1_nh, 1).to(device))


        #######################3
        # the second set of icnn
        ########################

        # first layer is treated separately
        self.icnn2_Wy0 = nn.Linear(self.icnn2_inputdim, self.icnn2_nh).to(device)

        self.icnn2_Wy_layers = []
        self.icnn2_Wz_layers = []
        for i in range(self.icnn2_num_layers-1):
            self.icnn2_Wy_layers.append(nn.Linear(self.icnn2_inputdim, self.icnn2_nh).to(device))
            self.icnn2_Wz_layers.append(PositiveLinear(self.icnn2_nh, self.icnn2_nh).to(device))

        # add final layer with output dimension = 1
        self.icnn2_Wy_layers.append(nn.Linear(self.icnn2_inputdim, 1).to(device))
        self.icnn2_Wz_layers.append(PositiveLinear(self.icnn2_nh, 1).to(device))


        #######################3
        # the third set of icnn
        ########################

        # first layer is treated separately
        # self.icnn3_Wy0 = nn.Linear(self.icnn3_inputdim, self.icnn3_nh).to(device)

        # self.icnn3_Wy_layers = []
        # self.icnn3_Wz_layers = []
        # for i in range(self.icnn3_num_layers-1):
        #     self.icnn3_Wy_layers.append(nn.Linear(self.icnn3_inputdim, self.icnn3_nh).to(device))
        #     self.icnn3_Wz_layers.append(PositiveLinear(self.icnn3_nh, self.icnn3_nh).to(device))

        # # add final layer with output dimension = 1
        # self.icnn3_Wy_layers.append(nn.Linear(self.icnn3_inputdim, 1).to(device))
        # self.icnn3_Wz_layers.append(PositiveLinear(self.icnn2_nh, 1).to(device))

        self.pred_linear = nn.Linear((args.dec_nh + args.nz), len(vocab))

        self.batchnorm = nn.BatchNorm1d(11)

        vocab_mask = torch.ones(len(vocab))
        # vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.trans_linear.weight)
        # model_init(self.pred_linear.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        # word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)

        c_init = self.trans_linear(torch.ones_like(z)).unsqueeze(0)
        # c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # print(output, z_)

        output = torch.cat((output, z_), -1)

        # output = self.batchnorm(output).cuda()

        # output = self.pred_linear(output)

        # output = self.batchnorm(output).cuda()

        # (batch_size * n_sample, seq_len, vocab_size)

        ##############
        # use ICNN
        ###############

        h1 = [[None] for i in range(self.icnn1_num_layers + 1)]
        h2 = [[None] for i in range(self.icnn2_num_layers + 1)]
        # h3 = [[None] for i in range(self.icnn3_num_layers + 1)]

        z_mid10 = output
        
        h1[0] = torch.pow((nn.LeakyReLU(0.2)(self.icnn1_Wy0(z_mid10))),2)

        # h1[0] = torch.pow((self.icnn1_Wy0(z_mid10)),2)

        # h1[0] = (self.icnn1_Wy0(torch.pow(z_mid10, 2) * 0.5)).sum(dim=-1)
        

        # h1[0] = (torch.pow(z_mid10, 2) * 0.5).sum(dim=-1)

        # print(h1[0].shape)

        # print(self.icnn1_Wy0.weight)

        for i in range(self.icnn1_num_layers):
            h1[i+1] = nn.LeakyReLU(0.2)(self.icnn1_Wz_layers[i](h1[i]) + self.icnn1_Wy_layers[i](z_mid10))

        # print(h1[-1].shape)

        dout_dx1 = torch.autograd.grad(h1[-1], [z_mid10], torch.ones_like(h1[-1]), create_graph=True)[0]


        # h_mid1 = dout_dx1

        # # print(self.icnn1_Wy0.weight)
        # # print(z_mid10)
        # # print(dout_dx1)

        # beta_icnn1_icnn2 = torch.eye(n=self.icnn1_inputdim, m=self.icnn2_inputdim).cuda()
        # h_mid2 = torch.matmul(dout_dx1, beta_icnn1_icnn2)

        h_mid2 = self.icnn1_icnn2(dout_dx1)
        

        z_mid20 = h_mid2

        h2[0] = (nn.LeakyReLU(0.2)(self.icnn2_Wy0(z_mid20)))**2
        for i in range(self.icnn2_num_layers):
            h2[i+1] = nn.LeakyReLU(0.2)(self.icnn2_Wz_layers[i](h2[i]) + self.icnn2_Wy_layers[i](z_mid20))
        dout_dx2 = torch.autograd.grad(h2[-1], [z_mid20], torch.ones_like(h2[-1]), create_graph=True)[0]

        # use additional layers if we want more ICNN and beta compositions

        # beta_icnn2_icnn3 = torch.eye(n=self.icnn2_inputdim, m=self.icnn3_inputdim).cuda()

        # h_mid3 = torch.matmul(dout_dx2, beta_icnn2_icnn3)

        icnn_output = dout_dx2

        # h_mid3 = self.icnn2_icnn3(dout_dx2)

        # z_mid30 = h_mid3

        # h3[0] = (nn.LeakyReLU(0.2)(self.icnn3_Wy0(z_mid30)))**2
        # for i in range(self.icnn3_num_layers):
        #     h3[i+1] = nn.LeakyReLU(0.2)(self.icnn3_Wz_layers[i](h3[i]) + self.icnn3_Wy_layers[i](z_mid30))
        # dout_dx3 = torch.autograd.grad(h3[-1], [z_mid30], torch.ones_like(h3[-1]), create_graph=True)[0]        

        # icnn_output = dout_dx3

        output_logits = icnn_output

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        # output_logits = self.decode(src, z)

        input = src


        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        # word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)

        c_init = self.trans_linear(torch.ones_like(z)).unsqueeze(0)
        # c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # print(output, z_)

        output = torch.cat((output, z_), -1)

        # output = self.batchnorm(output).cuda()

        # output = self.pred_linear(output)

        # output = self.batchnorm(output).cuda()

        # (batch_size * n_sample, seq_len, vocab_size)

        ##############
        # use ICNN
        ###############

        h1 = [[None] for i in range(self.icnn1_num_layers + 1)]
        h2 = [[None] for i in range(self.icnn2_num_layers + 1)]
        # h3 = [[None] for i in range(self.icnn3_num_layers + 1)]

        z_mid10 = output
        
        h1[0] = torch.pow((nn.LeakyReLU(0.2)(self.icnn1_Wy0(z_mid10))),2)

        # h1[0] = torch.pow((self.icnn1_Wy0(z_mid10)),2)

        # h1[0] = (self.icnn1_Wy0(torch.pow(z_mid10, 2) * 0.5)).sum(dim=-1)
        

        # h1[0] = (torch.pow(z_mid10, 2) * 0.5).sum(dim=-1)

        # print(h1[0].shape)

        # print(self.icnn1_Wy0.weight)

        for i in range(self.icnn1_num_layers):
            h1[i+1] = nn.LeakyReLU(0.2)(self.icnn1_Wz_layers[i](h1[i]) + self.icnn1_Wy_layers[i](z_mid10))

        # print(h1[-1].shape)

        dout_dx1 = torch.autograd.grad(h1[-1], [z_mid10], torch.ones_like(h1[-1]), create_graph=True)[0]


        # h_mid1 = dout_dx1

        # # print(self.icnn1_Wy0.weight)
        # # print(z_mid10)
        # # print(dout_dx1)


        # h_mid1 = dout_dx1
        # h_mid2 = self.icnn1_icnn2(h_mid1)

        # beta_icnn1_icnn2 = torch.eye(n=self.icnn1_inputdim, m=self.icnn2_inputdim).cuda()
        # h_mid2 = torch.matmul(dout_dx1, beta_icnn1_icnn2)

        h_mid2 = self.icnn1_icnn2(dout_dx1)
        

        z_mid20 = h_mid2

        h2[0] = (nn.LeakyReLU(0.2)(self.icnn2_Wy0(z_mid20)))**2
        for i in range(self.icnn2_num_layers):
            h2[i+1] = nn.LeakyReLU(0.2)(self.icnn2_Wz_layers[i](h2[i]) + self.icnn2_Wy_layers[i](z_mid20))
        dout_dx2 = torch.autograd.grad(h2[-1], [z_mid20], torch.ones_like(h2[-1]), create_graph=True)[0]

        # print(dout_dx1.shape, beta_icnn1_icnn2.shape, h_mid2.shape)

        # beta_icnn2_icnn3 = torch.eye(n=self.icnn2_inputdim, m=self.icnn3_inputdim).cuda()

        # h_mid3 = torch.matmul(dout_dx2, beta_icnn2_icnn3)

        icnn_output = dout_dx2

        # h_mid3 = self.icnn2_icnn3(dout_dx2)

        # z_mid30 = h_mid3

        # h3[0] = (nn.LeakyReLU(0.2)(self.icnn3_Wy0(z_mid30)))**2
        # for i in range(self.icnn3_num_layers):
        #     h3[i+1] = nn.LeakyReLU(0.2)(self.icnn3_Wz_layers[i](h3[i]) + self.icnn3_Wy_layers[i](z_mid30))
        # dout_dx3 = torch.autograd.grad(h3[-1], [z_mid30], torch.ones_like(h3[-1]), create_graph=True)[0]        

        # icnn_output = dout_dx3

        output_logits = icnn_output

        # output = torch.cat((output, icnn_output), -1)
        # output = torch.cat((output, z_), -1)

        # output = output + icnn_output

        # output_logits = dout_dx1

        # output_logits = self.pred_linear(dout_dx1)


        # output_logits = output
       
        # output_logits = self.pred_linear(icnn_output)


        # output_logits = torch.matmul(output, 0.01*torch.ones(1, self.nh + self.nz, len(self.vocab)).cuda())

        # print(output_logits)



        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)

        # S = torch.svd(torch.triu(self.icnn1_icnn2.weight))[1]
        # rank_penalty = (torch.min(torch.log(S),1e-6*torch.ones_like(S))).sum()
        # torch.log(1. - self.icnn1_inputdim + torch.matrix_rank(self.icnn1_icnn2.weight))
        # print(rank_penalty)

        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)


    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)



class LSTMDecoder(DecoderBase):
    """LSTM decoder with constant-length batching"""
    '''
    This is the original vanilla LSTMDecoder implemented by 
    https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/decoders/dec_lstm.py
    '''
    def __init__(self, args, vocab, model_init, emb_init):
        super(LSTMDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=args.ni + args.nz,
                            hidden_size=args.dec_nh,
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(args.dec_nh, len(vocab), bias=False)

        vocab_mask = torch.ones(len(vocab))
        # vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.trans_linear.weight)
        # model_init(self.pred_linear.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)


    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)

    def beam_search_decode(self, z, K=5):
        """beam search decoding, code is based on
        https://github.com/pcyin/pytorch_basic_nmt/blob/master/nmt.py
        the current implementation decodes sentence one by one, further batching would improve the speed
        Args:
            z: (batch_size, nz)
            K: the beam width
        Returns: List1
            List1: the decoded word sentence list
        """

        decoded_batch = []
        batch_size, nz = z.size()

        # (1, batch_size, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            # Start with the start of the sentence token
            decoder_input = torch.tensor([[self.vocab["<s>"]]], dtype=torch.long, device=self.device)
            decoder_hidden = (h_init[:,idx,:].unsqueeze(1), c_init[:,idx,:].unsqueeze(1))

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0., 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < 100:
                t += 1

                # (len(live), 1)
                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=0)

                # (1, len(live), nh)
                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)


                # (len(live), 1, ni) --> (len(live), 1, ni+nz)
                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    len(live_hypotheses), 1, nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                # (len(live), 1, vocab_size)
                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses], dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(len(live_hypotheses), 1, 1)

                # (len(live) * vocab_size)
                decoder_output = decoder_output.view(-1)

                # (K)
                log_prob, indexes = torch.topk(decoder_output, K-len(completed_hypotheses))

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.id2word(n.wordid.item()))
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(self.vocab.id2word(n.wordid.item()))

                utterance = utterance[::-1]

                utterances.append(utterance)

                # only save the top 1
                break

            decoded_batch.append(utterances[0])

        return decoded_batch

    def greedy_decode(self, z):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            max_index = torch.argmax(output_logits, dim=1)
            # max_index = torch.multinomial(probs, num_samples=1)

            decoder_input = max_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(max_index[i].item()))

            mask = torch.mul((max_index != end_symbol), mask)

        return decoded_batch

    def sample_decode(self, z):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            sample_prob = F.softmax(output_logits, dim=1)
            sample_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = sample_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(sample_index[i].item()))

            mask = torch.mul((sample_index != end_symbol), mask)

        return decoded_batch
