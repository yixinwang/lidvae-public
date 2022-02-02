import sys
import os
import time
import importlib
import argparse
import itertools
from datetime import datetime

from argparse import Namespace

import numpy as np

import torch
from torch import nn, optim

from importlib import reload  

from data import MonoTextData

# import modules
# reload(modules)

from modules import VAE
from modules import dec_lstm
from modules import GaussianLSTMEncoder, LSTMEncoder, idLSTMDecoder, LSTMDecoder
from logger import Logger

from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_mi_li_et_al, calc_mi, calc_au, sample_sentences, reconstruct, reconstruct_li_et_al, visualize_latent_li_et_al, test, test_li_et_al_ae, test_li_et_al_vae
from exp_utils import create_exp_dir

logging = None

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, default='yahoo', help='dataset to use', choices=['yahoo', 'yelp', 'synthetic'])

    parser.add_argument('--model', type=str, default='idvae', help='model to use', choices=['idvae', 'vae', 'laggingvae', 'betavae', 'li_et_al'])

    parser.add_argument('--betavae', type=float, default=0.2, help='the beta value of betavae')

    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--nsamples', type=int, default=1, help='number of iw samples for training')
    parser.add_argument('--subsample', type=int, default=30000, help='subsample training data')
    parser.add_argument('--subsample_eval', type=int, default=10000, help='subsample test / val data')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')


    # decoding
    parser.add_argument('--decode_from', type=str, default="", help="pretrained model path")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--decode_input', type=str, default="", help="input text file to perform reconstruction")


    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=0, help="number of annealing epochs")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")

    # inference parameters
    parser.add_argument('--aggressive', type=int, default=0,
                         help='apply aggressive training when nonzero, reduce to vanilla VAE when aggressive is 0')
    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--l2reg', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--optim', type=str, default="adam", help='optimizer')

    # parameters for Li et al (2019) AE paramters
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')
    parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--iw_train_nsamples', type=int, default=-1)
    parser.add_argument('--iw_train_ns', type=int, default=1, help='number of iw samples for training in each batch')
    parser.add_argument("--fix_var", type=float, default=-1)
    parser.add_argument("--freeze_epoch", type=int, default=-1)
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--load_best_epoch", type=int, default=15)

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                         help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=0,
                        help="save checkpoint every epoch before this number")
    parser.add_argument("--save_latent", type=int, default=0)

    # parameters for Li et al (2019) VAE paramters
    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "models/%s" % args.dataset
    log_dir = "logs/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]

    id_ = "%s_%s_aggressive%d_kls%.2f_lr%.2f_l2reg%.2f_warm%d_%d_%d_%d" % \
            (args.dataset, args.model, args.aggressive, args.kl_start, args.lr, args.l2reg,
             args.warm_up, args.jobid, args.taskid, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path
    print("save path", args.save_path)

    args.log_path = os.path.join(log_dir, id_ + ".log")
    print("log path", args.log_path)

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)


    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args

def subsample_data(train_data, nsample):

    index_arr = np.arange(len(train_data.data))
    # np.random.shuffle(index_arr)

    batch_ids = index_arr[: nsample]
    batch_data = [train_data.data[index] for index in batch_ids]

    # uncomment this line if the dataset has variable length
    batch_data.sort(key=lambda e: -len(e))

    return batch_data

def main_idvae(args):

    if args.cuda:
        print('using cuda')

    # args.nz=5
    # args.lr=1e-2
    # args.batch_size = 16
    # args.epochs = 5000000
    # args.l2reg = 0.000

    # args.icnn_num_layers = 2
    # args.icnn_nh = 128

    timenow = int(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))

    outfilename = 'out_'+args.dataset+'_textidvae'+str(timenow)

    for outfile in [outfilename+'.log']:
        with open(outfile, 'a') as f:
            f.write("hi i'm starting\n\n") 
            f.write(str(args))
            f.write("\n\n")

    print(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    opt_dict = {"not_improved": 0, "lr": args.lr, "best_loss": 1e4}

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)


    # subsample training data

    nsample = args.subsample

    evalsample = args.subsample_eval

    train_data.data = subsample_data(train_data, nsample)
    val_data.data = subsample_data(val_data, evalsample)
    test_data.data = subsample_data(test_data, evalsample)



    for outfile in [outfilename+'.log']:
        with open(outfile, 'a') as f:
            f.write('ID-sequential-VAE') 
            f.write('Train data: ' + str(len(train_data)) + ' samples')
            f.write('finish reading datasets, vocab size is ' + str(len(vocab)))
            # f.write('dropped sentences: ' + str(train_data.dropped))
            f.write("\n\n")

    print('ID-sequential-VAE')

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    # print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    # log_niter = 1

    model_init = uniform_initializer(0.001)
    emb_init = uniform_initializer(0.001)

    # model_init = xavier_normal_initializer(0.0)
    # emb_init = xavier_normal_initializer(0.0)


    if args.enc_type == 'lstm':
        encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")


    decoder = idLSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(device)


    decoder_pars = list(vae.decoder.parameters()) + \
        list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn1_Wy_layers])) + \
        list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn1_Wz_layers])) + \
        list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn2_Wy_layers])) + \
        list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn2_Wz_layers]))

        #  + \
        # list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn3_Wy_layers])) + \
        # list(itertools.chain.from_iterable([list(layer.parameters()) for layer in vae.decoder.icnn3_Wz_layers]))

    if args.optim == "adam":
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg)

        dec_optimizer = optim.Adam(decoder_pars, lr=args.lr, weight_decay=args.l2reg)
    elif args.optim == "rmsprop":
        enc_optimizer = optim.RMSprop(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg)

        dec_optimizer = optim.RMSprop(decoder_pars, lr=args.lr, weight_decay=args.l2reg)    
    elif args.optim == "sgd":
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg, momentum=args.momentum)

        dec_optimizer = optim.SGD(decoder_pars, lr=args.lr, weight_decay=args.l2reg, momentum=args.momentum)   
           
    # enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2reg)
    # dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2reg)



    opt_dict['lr'] = args.lr

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    aggressive_flag = True if args.aggressive else False
    vae.train()
    start = time.time()

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)
    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        for i in np.random.permutation(len(train_data_batch)):
            batch_data = train_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size

            report_num_sents += batch_size

            kl_weight = 1.0
            # print('kl_weight', kl_weight)

            sub_iter = 1
            batch_data_enc = batch_data
            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while aggressive_flag and sub_iter < 100:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                burn_batch_size, burn_sents_len = batch_data_enc.size()
                burn_num_words += (burn_sents_len - 1) * burn_batch_size

                loss, loss_rc, loss_kl = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)

                burn_cur_loss += loss.sum().item()
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.random_integers(0, len(train_data_batch) - 1)

                batch_data_enc = train_data_batch[id_]

                if sub_iter % 15 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

                sub_iter += 1

                # if sub_iter >= 30:
                #     break

            # print(sub_iter)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()


            loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)

            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            if not aggressive_flag:
                enc_optimizer.step()

            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            # print("iter", iter_)
            if iter_ % log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                # if aggressive_flag or epoch == 0:
                vae.eval()
                with torch.no_grad():
                    mi = calc_mi(vae, val_data_batch)
                    au, au_var = calc_au(vae, val_data_batch)

                vae.train()

                for outfile in [outfilename+'.log']:
                    with open(outfile, 'a') as f:
                        f.write('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                           'au %d, time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                           report_rec_loss / report_num_sents, au, time.time() - start)) 
                        f.write("au_var" + str(au_var))

                print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                       'au %d, time elapsed %.2fs' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                       report_rec_loss / report_num_sents, au, time.time() - start))
                print(au_var)

                # else:
                #     print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                #            'time elapsed %.2fs' %
                #            (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                #            report_rec_loss / report_num_sents, time.time() - start))

                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            if (iter_+1) % (100*log_niter) == 0:

                # with torch.no_grad():
                args_tmp = Namespace(**vars(args))
                args_tmp.iw_nsamples = 5
                loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args_tmp)
                with torch.no_grad():
                    au, au_var = calc_au(vae, val_data_batch)

                for outfile in [outfilename+'.log']:
                    with open(outfile, 'a') as f:
                        f.write("%d active units" % au)
                        f.write('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
                        f.write('mi: %.4f, kl: %.4f' % (mi, kl))

                print("%d active units" % au)
                print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
                print('mi: %.4f, kl: %.4f' % (mi, kl))



            iter_ += 1

            # if aggressive_flag and (iter_ % len(train_data_batch)) == 0:
            #     vae.eval()
            #     cur_mi = calc_mi(vae, val_data_batch)
            #     vae.train()

            #     for outfile in [outfilename+'.log']:
            #         with open(outfile, 'a') as f:
            #             f.write("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))

            #     print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
            #     if cur_mi - pre_mi < 0:
            #         aggressive_flag = False

            #         for outfile in [outfilename+'.log']:
            #             with open(outfile, 'a') as f:
            #                 f.write("STOP BURNING")

            #         print("STOP BURNING")

            #     pre_mi = cur_mi

        for outfile in [outfilename+'.log']:
            with open(outfile, 'a') as f:
                f.write('kl weight %.4f' % kl_weight)
        print('kl weight %.4f' % kl_weight)

        vae.eval()

        # with torch.no_grad():
        loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
        with torch.no_grad():
            au, au_var = calc_au(vae, val_data_batch)

        for outfile in [outfilename+'.log']:
            with open(outfile, 'a') as f:
                f.write("%d active units" % au)
                f.write('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
                f.write('mi: %.4f, kl: %.4f' % (mi, kl))

        print("%d active units" % au)
        print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
        print('mi: %.4f, kl: %.4f' % (mi, kl))


            # print(au_var)

        if loss < best_loss:
            for outfile in [outfilename+'.log']:
                with open(outfile, 'a') as f:
                    f.write('update best loss')
            print('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            best_ppl = ppl
            torch.save(vae.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >=15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))

                for outfile in [outfilename+'.log']:
                    with open(outfile, 'a') as f:
                        f.write('new lr: %f' % opt_dict["lr"])
                   
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                # enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                # dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

                if args.optim == "adam":
                    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg)

                    dec_optimizer = optim.Adam(decoder_pars, lr=args.lr, weight_decay=args.l2reg)
                elif args.optim == "rmsprop":
                    enc_optimizer = optim.RMSprop(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg)

                    dec_optimizer = optim.RMSprop(decoder_pars, lr=args.lr, weight_decay=args.l2reg)    
                elif args.optim == "sgd":
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, weight_decay=args.l2reg, momentum=args.momentum)

                    dec_optimizer = optim.SGD(decoder_pars, lr=args.lr, weight_decay=args.l2reg, momentum=args.momentum)   
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        # if epoch % args.test_nepoch == 0:
        #     # with torch.no_grad():
        #     loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

        if iter_ % args.test_nepoch * 100 == 0:
            # with torch.no_grad():
            loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()


    loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

    with torch.no_grad():
        au, au_var = calc_au(vae, test_data_batch)

    for outfile in [outfilename+'.log']:
        with open(outfile, 'a') as f:
            f.write("%d active units" % au)

    print("%d active units" % au)

        # print(au_var)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    # with torch.no_grad():
    calc_iwnll(vae, test_data_batch, args)


def main_baseline(args):


    timenow = int(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))

    outfilename = 'out_'+args.dataset+'_textvae_'+args.model+str(timenow)

    for outfile in [outfilename+'.log']:
        with open(outfile, 'a') as f:
            f.write("hi i'm starting\n\n") 
            f.write(str(args))
            f.write("\n\n")

    print(args)

    if args.cuda:
        print('using cuda')

    print(args)

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    # subsample training data

    nsample = args.subsample

    evalsample = args.subsample_eval

    train_data.data = subsample_data(train_data, nsample)
    val_data.data = subsample_data(val_data, evalsample)
    test_data.data = subsample_data(test_data, evalsample)


    for outfile in [outfilename+'.log']:
        with open(outfile, 'a') as f:
            f.write('sequential-VAE'+args.model) 
            f.write('Train data: ' + str(len(train_data)) + ' samples')
            f.write('finish reading datasets, vocab size is ' + str(len(vocab)))
            # f.write('dropped sentences: ' + str(train_data.dropped))
            f.write("\n\n")

    print('sequential-VAE'+args.model)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()


    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.enc_type == 'lstm':
        encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(device)

    if args.decode_from != "":
        print('begin decoding')
        vae.load_state_dict(torch.load(args.decode_from))
        vae.eval()
        save_dir = "samples/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = ".".join(args.decode_from.split("/")[-1].split(".")[:-1]) + \
                "_{}".format(args.decoding_strategy)
        with torch.no_grad():
            if args.decode_input != "":
                decode_data = MonoTextData(args.decode_input, vocab=vocab)

                reconstruct(vae, decode_data, args.decoding_strategy,
                    os.path.join(save_dir, path + ".rec"), args.device)
            else:
                z = vae.sample_from_prior(100)
                sample_from_prior(vae, z, args.decoding_strategy,
                    os.path.join(save_dir, path + ".sample"))

        return


    if args.eval:
        print('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test(vae, test_data_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            print("%d active units" % au)
            print(au_var)

            for outfile in [outfilename+'.log']:
                with open(outfile, 'a') as f:
                    f.write("%d active units" % au)
                    f.write("au_var" + str(au_var))

            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            calc_iwnll(vae, test_data_batch, args)

        return

    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0, momentum=args.momentum)
    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0, momentum=args.momentum)
    opt_dict['lr'] = 1.0

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    aggressive_flag = True if args.aggressive else False
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (1e-8 + args.warm_up * (len(train_data) / args.batch_size))

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)
    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        for i in np.random.permutation(len(train_data_batch)):
            batch_data = train_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size

            report_num_sents += batch_size

            # kl_weight = 1.0
            kl_weight = min(1.0, kl_weight + anneal_rate)
            if args.model == 'betavae':
                kl_weight = args.betavae

            # print('kl_weight', kl_weight)

            sub_iter = 1
            batch_data_enc = batch_data
            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while aggressive_flag and sub_iter < 100:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                burn_batch_size, burn_sents_len = batch_data_enc.size()
                burn_num_words += (burn_sents_len - 1) * burn_batch_size

                loss, loss_rc, loss_kl = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)

                burn_cur_loss += loss.sum().item()
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.random_integers(0, len(train_data_batch) - 1)

                batch_data_enc = train_data_batch[id_]

                if sub_iter % 15 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

                sub_iter += 1

                # if sub_iter >= 30:
                #     break

            # print(sub_iter)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()


            loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)

            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            if not aggressive_flag:
                enc_optimizer.step()

            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            # print("iter", iter_)
            if iter_ % log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                if aggressive_flag or epoch == 0:
                    vae.eval()
                    with torch.no_grad():
                        mi = calc_mi(vae, val_data_batch)
                        au, _ = calc_au(vae, val_data_batch)
                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                           'au %d, time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                           report_rec_loss / report_num_sents, au, time.time() - start))
                    for outfile in [outfilename+'.log']:
                        with open(outfile, 'a') as f:
                            f.write('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                               'au %d, time elapsed %.2fs' %
                               (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                               report_rec_loss / report_num_sents, au, time.time() - start)) 
                            # f.write("au_var" + str(au_var))
                else:
                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start))
                    for outfile in [outfilename+'.log']:
                        with open(outfile, 'a') as f:
                            f.write('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                               'time elapsed %.2fs' %
                               (epoch, iter_, train_loss, report_kl_loss / report_num_sents, 
                               report_rec_loss / report_num_sents, time.time() - start)) 
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            iter_ += 1

            if aggressive_flag and (iter_ % len(train_data_batch)) == 0:
                vae.eval()
                cur_mi = calc_mi(vae, val_data_batch)
                vae.train()
                print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
                for outfile in [outfilename+'.log']:
                    with open(outfile, 'a') as f:
                        f.write("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))

                if cur_mi - pre_mi < 0:
                    aggressive_flag = False
                    print("STOP BURNING")
                    for outfile in [outfilename+'.log']:
                        with open(outfile, 'a') as f:
                            f.write("STOP BURNING")
                pre_mi = cur_mi

        print('kl weight %.4f' % kl_weight)
        for outfile in [outfilename+'.log']:
            with open(outfile, 'a') as f:
                f.write('kl weight %.4f' % kl_weight)

        vae.eval()
        with torch.no_grad():
            loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
            au, au_var = calc_au(vae, val_data_batch)
            print("%d active units" % au)
            print(au_var)
            for outfile in [outfilename+'.log']:
                with open(outfile, 'a') as f:
                    f.write("%d active units" % au)
                    f.write("au_var" + str(au_var))

        if loss < best_loss:
            print('update best loss')
            for outfile in [outfilename+'.log']:
                with open(outfile, 'a') as f:
                    f.write('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            best_ppl = ppl
            torch.save(vae.state_dict(), args.save_path)


        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >=15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                print('new lr: %f' % opt_dict["lr"])

                for outfile in [outfilename+'.log']:
                    with open(outfile, 'a') as f:
                        f.write('new lr: %f' % opt_dict["lr"])

            for outfile in [outfilename+'.log']:
                with open(outfile, 'a') as f:
                    f.write('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        # if epoch % args.test_nepoch == 0:
        #     # with torch.no_grad():
        #     loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

        if iter_ % args.test_nepoch * 100 == 0:
            # with torch.no_grad():
            loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)


        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)
        print("%d active units" % au)
        print(au_var)
        for outfile in [outfilename+'.log']:
            with open(outfile, 'a') as f:
                f.write("%d active units" % au)
                f.write("au_var" + str(au_var))

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)

def main_li_et_al_ae(args):
    global logging
    debug = (args.reconstruct_from != "" or args.eval == True) # don't make exp dir for reconstruction
    logging = create_exp_dir(args.exp_dir, scripts_to_save=None, debug=debug)

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e16}

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)


    # subsample training data

    nsample = args.subsample

    evalsample = args.subsample_eval

    train_data.data = subsample_data(train_data, nsample)
    val_data.data = subsample_data(val_data, evalsample)
    test_data.data = subsample_data(test_data, evalsample)


    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    #sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device

    if args.enc_type == 'lstm':
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path)
        #curr_state_dict = vae.state_dict()
        #curr_state_dict.update(loaded_state_dict)
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

        if args.reset_dec:
            vae.decoder.reset_parameters(model_init, emb_init)


    if args.eval:
        logging('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test_li_et_al_ae(vae, test_data_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            logging("%d active units" % au)
            # print(au_var)

            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)

            nll, ppl = calc_iwnll(vae, test_data_batch, args)
            logging('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))

        return

    if args.reconstruct_from != "":
        print("begin decoding")
        sys.stdout.flush()

        vae.load_state_dict(torch.load(args.reconstruct_from))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)
            # test_li_et_al_ae(vae, test_data_batch, "TEST", args)
            reconstruct_li_et_al(vae, test_data_batch, vocab, args.decoding_strategy, args.reconstruct_to)

        return

    if args.opt == "sgd":
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, momentum=args.momentum)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001)
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001)
        opt_dict['lr'] = 0.001
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    vae.train()
    start = time.time()

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.epochs):
            report_kl_loss = report_rec_loss = report_loss = 0
            report_num_words = report_num_sents = 0

            for i in np.random.permutation(len(train_data_batch)):

                batch_data = train_data_batch[i]
                batch_size, sent_len = batch_data.size()

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                kl_weight = args.beta

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                if args.iw_train_nsamples < 0:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                else:
                    loss, loss_rc, loss_kl = vae.loss_iw(batch_data, kl_weight, nsamples=args.iw_train_nsamples, ns=ns)
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()

                enc_optimizer.step()
                dec_optimizer.step()

                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()
                report_loss += loss.item() * batch_size

                if iter_ % log_niter == 0:
                    #train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                    train_loss = report_loss / report_num_sents
                    logging('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs, kl_weight %.4f' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start, kl_weight))

                    #sys.stdout.flush()

                    report_rec_loss = report_kl_loss = report_loss = 0
                    report_num_words = report_num_sents = 0

                iter_ += 1

            logging('kl weight %.4f' % kl_weight)

            vae.eval()
            with torch.no_grad():
                loss, nll, kl, ppl, mi = test_li_et_al_ae(vae, val_data_batch, "VAL", args)
                au, au_var = calc_au(vae, val_data_batch)
                logging("%d active units" % au)
                # print(au_var)

            if args.save_ckpt > 0 and epoch <= args.save_ckpt:
                logging('save checkpoint')
                torch.save(vae.state_dict(), os.path.join(args.exp_dir, f'model_ckpt_{epoch}.pt'))

            if loss < best_loss:
                logging('update best loss')
                best_loss = loss
                best_nll = nll
                best_kl = kl
                best_ppl = ppl
                torch.save(vae.state_dict(), args.save_path)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch and epoch >=args.load_best_epoch:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    vae.load_state_dict(torch.load(args.save_path))
                    logging('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            if decay_cnt == max_decay:
                break

            if epoch % args.test_nepoch == 0:
                with torch.no_grad():
                    loss, nll, kl, ppl, _ = test_li_et_al_ae(vae, test_data_batch, "TEST", args)

            if args.save_latent > 0 and epoch <= args.save_latent:
                visualize_latent_li_et_al(args, epoch, vae, "cuda", test_data)

            vae.train()

    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test_li_et_al_ae(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)
        logging("%d active units" % au)
        # print(au_var)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        nll, ppl = calc_iwnll(vae, test_data_batch, args)
        logging('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))


def main_li_et_al_vae(args):
    global logging
    debug = (args.reconstruct_from != "" or args.eval == True) # don't make exp dir for reconstruction
    logging = create_exp_dir(args.exp_dir, scripts_to_save=None, debug=debug)

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)


    # subsample training data

    nsample = args.subsample

    evalsample = args.subsample_eval

    train_data.data = subsample_data(train_data, nsample)
    val_data.data = subsample_data(val_data, evalsample)
    test_data.data = subsample_data(test_data, evalsample)



    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    #sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device

    if args.enc_type == 'lstm':
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path)
        #curr_state_dict = vae.state_dict()
        #curr_state_dict.update(loaded_state_dict)
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

        if args.reset_dec:
            logging("\n-------reset decoder-------\n")
            vae.decoder.reset_parameters(model_init, emb_init)


    if args.eval:
        logging('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test_li_et_al_vae(vae, test_data_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            logging("%d active units" % au)
            # print(au_var)

            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            nll, ppl = calc_iwnll(vae, test_data_batch, args)
            logging('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))

        return

    if args.reconstruct_from != "":
        print("begin decoding")
        vae.load_state_dict(torch.load(args.reconstruct_from))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)
            # test(vae, test_data_batch, "TEST", args)
            reconstruct_li_et_al(vae, test_data_batch, vocab, args.decoding_strategy, args.reconstruct_to)

        return

    if args.opt == "sgd":
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, momentum=args.momentum)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001)
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001)
        opt_dict['lr'] = 0.001
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    if args.warm_up > 0:
        anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    else:
        anneal_rate = 0

    dim_target_kl = args.target_kl / float(args.nz)

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.epochs):
            report_kl_loss = report_rec_loss = report_loss = 0
            report_num_words = report_num_sents = 0

            for i in np.random.permutation(len(train_data_batch)):

                batch_data = train_data_batch[i]
                batch_size, sent_len = batch_data.size()

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                kl_weight = min(1.0, kl_weight + anneal_rate)

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                
                if args.fb == 0:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                elif args.fb == 1:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                    kl_mask = (loss_kl > args.target_kl).float()
                    loss = loss_rc + kl_mask * kl_weight * loss_kl 
                elif args.fb == 2:
                    mu, logvar = vae.encoder(batch_data)
                    z = vae.encoder.reparameterize(mu, logvar, args.nsamples)
                    loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                    kl_mask = (loss_kl > dim_target_kl).float()
                    fake_loss_kl = (kl_mask * loss_kl).sum(dim=1)
                    loss_rc = vae.decoder.reconstruct_error(batch_data, z).mean(dim=1)
                    loss = loss_rc + kl_weight * fake_loss_kl 
                elif args.fb == 3:
                    loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                    kl_mask = (loss_kl.mean() > args.target_kl).float()
                    loss = loss_rc + kl_mask * kl_weight * loss_kl 

                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()

                enc_optimizer.step()
                dec_optimizer.step()

                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()
                report_loss += loss_rc.item() + loss_kl.item()

                if iter_ % log_niter == 0:
                    #train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                    train_loss = report_loss / report_num_sents

                    logging('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time %.2fs, kl_weight %.4f' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start, kl_weight))

                    #sys.stdout.flush()

                    report_rec_loss = report_kl_loss = report_loss = 0
                    report_num_words = report_num_sents = 0

                iter_ += 1

            logging('kl weight %.4f' % kl_weight)
            logging('lr {}'.format(opt_dict["lr"]))

            vae.eval()
            with torch.no_grad():
                loss, nll, kl, ppl, mi = test_li_et_al_vae(vae, val_data_batch, "VAL", args)
                au, au_var = calc_au(vae, val_data_batch)
                logging("%d active units" % au)
                # print(au_var)

            if args.save_ckpt > 0 and epoch <= args.save_ckpt:
                logging('save checkpoint')
                torch.save(vae.state_dict(), os.path.join(args.exp_dir, f'model_ckpt_{epoch}.pt'))

            if loss < best_loss:
                logging('update best loss')
                best_loss = loss
                best_nll = nll
                best_kl = kl
                best_ppl = ppl
                torch.save(vae.state_dict(), args.save_path)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch and epoch >= args.load_best_epoch:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    vae.load_state_dict(torch.load(args.save_path))
                    logging('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            if decay_cnt == max_decay:
                break

            if epoch % args.test_nepoch == 0:
                with torch.no_grad():
                    loss, nll, kl, ppl, _ = test_li_et_al_vae(vae, test_data_batch, "TEST", args)

            if args.save_latent > 0 and epoch <= args.save_latent:
                visualize_latent_li_et_al(args, epoch, vae, "cuda", test_data)

            vae.train()

    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test_li_et_al_vae(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)
        logging("%d active units" % au)
        # print(au_var)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        nll, ppl = calc_iwnll(vae, test_data_batch, args)
        logging('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))

if __name__ == '__main__':
    args = init_config()
    clip_grad = 5.0
    decay_epoch = 2
    if args.decode_from == "" and not args.eval:
        sys.stdout = Logger(args.log_path)
    if args.model == 'idvae':
        args.aggressive = 0
        lr_decay = 0.2
        max_decay = 500000
        main_idvae(args)
    else:
        lr_decay = 0.5
        max_decay = 5
        if args.model == 'vae':
            args.aggressive = 0
            main_baseline(args)
        elif args.model == 'laggingvae':
            args.aggressive = 1
            args.warm_up = 10
            args.kl_start = 0.1
            main_baseline(args)
        elif args.model == 'betavae':
            args.aggressive = 0
            main_baseline(args)
        elif args.model == 'li_et_al':
            # set load and save paths
            load_str = "_load" if args.load_path != "" else ""
            iw_str = "_iw{}".format(args.iw_train_nsamples) if args.iw_train_nsamples > 0 else ""

            if args.exp_dir == None:
                args.exp_dir = "exp_{}_beta/{}_lr{}_beta{}_drop{}_{}".format(
                    args.dataset, args.dataset, args.lr, args.beta, args.dec_dropout_in, iw_str)

            if len(args.load_path) <= 0 and args.eval:
                args.load_path = os.path.join(args.exp_dir, 'model.pt')

            args.save_path = os.path.join(args.exp_dir, 'model.pt')

            decay_epoch = 5
            ns=2
            logging = None
            args.beta = 0.0
            args.lr = 0.5
            # first fit ae with Li et al (2019)
            ae_exp_dir=args.exp_dir
            main_li_et_al_ae(args)
            # next fit vae with Li et al (2019)
            args.load_path = os.path.join(ae_exp_dir, 'model.pt')
            args.reset_dec = True
            args.kl_start = 0
            args.warm_up = 0
            args.target_kl = 8
            args.fb = 2
            args.lr = 0.5

            load_str = "_load" if args.load_path != "" else ""
            if args.fb == 0:
                fb_str = ""
            elif args.fb == 1:
                fb_str = "_fb"
            elif args.fb == 2:
                fb_str = "_fbdim"
            elif args.fb == 3:
                fb_str = "_fb3"
            args.exp_dir = "exp_{}{}/{}_warm{}_kls{:.1f}{}_tr{}".format(args.dataset,
                    load_str, args.dataset, args.warm_up, args.kl_start, fb_str, args.target_kl)
            if len(args.load_path) <= 0 and args.eval:
                args.load_path = os.path.join(args.exp_dir, 'model.pt')
            main_li_et_al_vae(args)
