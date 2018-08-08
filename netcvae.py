"""NetCVAE model on modified KNN_IHDP
"""

from __future__ import absolute_import
from __future__ import division

import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar

from datasets import KNN_IHDP
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

from utils import fc_net, get_y0_y1, nodeidx2map, average_y, map_concatenate, merge_scores
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=2)
parser.add_argument('-earl', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-print_every', type=int, default=1)
args = parser.parse_args()

#what does this do?
args.true_post = True


dataset = KNN_IHDP(replications=args.reps)
dimx = 25
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None # batch size during training
d = 20  # latent dimension
lamba = 1e-5  # weight decay
nh, h = 6, 400  # number and size of hidden layers
alpha = 0.005 # influence from neighbor

for it, (train_i, valid_i, test_i, train_j, valid_j, test_j, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(it + 1, args.reps))
    (nodei_idx_tr, xitr, titr, yitr), (yi_cftr, mui0tr, mui1tr) = train_i
    (nodei_idx_va, xiva, tiva, yiva), (yi_cfva, mui0va, mui1va) = valid_i
    (nodei_idx_te, xite, tite, yite), (yi_cfte, mui0te, mui1te) = test_i

    (nodej_idx_tr, xjtr, tjtr, yjtr), (yj_cftr, muj0tr, muj1tr) = train_j
    (nodej_idx_va, xjva, tjva, yjva), (yj_cfva, muj0va, muj1va) = valid_j
    (nodej_idx_te, xjte, tjte, yjte), (yj_cfte, muj0te, muj1te) = test_j

    # we need to get the position of each given node idx

    nodei_map_tr, unipos_i_tr = nodeidx2map(nodei_idx_tr)
    nodei_map_va, unipos_i_va = nodeidx2map(nodei_idx_va)
    nodei_map_te, unipos_i_te = nodeidx2map(nodei_idx_te)

    pos_i_tr, pos_i_va, pos_i_te = np.asarray(list(unipos_i_tr.values())), np.asarray(list(unipos_i_va.values())), np.asarray(list(unipos_i_te.values()))

    nodej_map_tr, unipos_j_tr = nodeidx2map(nodej_idx_tr)
    nodej_map_va, unipos_j_va = nodeidx2map(nodej_idx_va)
    nodej_map_te, unipos_j_te = nodeidx2map(nodej_idx_te)

    pos_j_tr, pos_j_va, pos_j_te = np.asarray(list(unipos_j_tr.values())), np.asarray(list(unipos_j_va.values())), np.asarray(list(unipos_j_te.values()))

    #pos_i_alltr = np.concatenate([pos_i_tr,pos_i_va+len(nodei_idx_tr)],axis=0)
    #pos_j_alltr = np.concatenate([pos_j_tr,pos_j_va+len(nodej_idx_tr)],axis=0)

    evaluatori_test = Evaluator(yite[pos_i_te], tite[pos_i_te], y_cf=yi_cfte[pos_i_te], mu0=mui0te[pos_i_te], mu1=mui1te[pos_i_te])
    evaluatorj_test = Evaluator(yjte[pos_j_te], tjte[pos_j_te], y_cf=yj_cfte[pos_j_te], mu0=muj0te[pos_j_te], mu1=muj1te[pos_j_te])

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    
    xitr, xiva, xite = xitr[:, perm], xiva[:, perm], xite[:, perm]
    xjtr, xjva, xjte = xjtr[:, perm], xjva[:, perm], xjte[:, perm]

    nodei_idx_alltr, xialltr, tialltr, yialltr = np.concatenate([nodei_idx_tr, nodei_idx_va],axis=0), np.concatenate([xitr, xiva], axis=0), np.concatenate([titr, tiva], axis=0), np.concatenate([yitr, yiva], axis=0)
    nodej_idx_alltr, xjalltr, tjalltr, yjalltr = np.concatenate([nodej_idx_tr, nodej_idx_va],axis=0), np.concatenate([xjtr, xjva], axis=0), np.concatenate([tjtr, tjva], axis=0), np.concatenate([yjtr, yjva], axis=0)

    nodei_map_alltr, unipos_i_alltr = nodeidx2map(nodei_idx_alltr)
    nodej_map_alltr, unipos_j_alltr = nodeidx2map(nodej_idx_alltr)

    # pos of the unique nodes in all pairs as node i or node j

    pos_i_alltr = np.asarray(list(unipos_i_alltr.values()))
    pos_j_alltr = np.asarray(list(unipos_j_alltr.values()))
    
    evaluatori_train = Evaluator(yialltr[pos_i_alltr], tialltr[pos_i_alltr], y_cf=np.concatenate([yi_cftr, yi_cfva], axis=0)[pos_i_alltr],
                                mu0=np.concatenate([mui0tr, mui0va], axis=0)[pos_i_alltr], mu1=np.concatenate([mui1tr, mui1va], axis=0)[pos_i_alltr])

    evaluatorj_train = Evaluator(yjalltr[pos_j_alltr], tjalltr[pos_j_alltr], y_cf=np.concatenate([yj_cftr, yj_cfva], axis=0)[pos_j_alltr],
                                mu0=np.concatenate([muj0tr, muj0va], axis=0)[pos_j_alltr], mu1=np.concatenate([muj1tr, muj1va], axis=0)[pos_j_alltr])

    # zero mean, unit variance for y during training
    
    # treat i and j separately
    # yim, yis = np.mean(yitr), np.std(yitr)
    # yitr, yiva = (yitr - yim) / yis, (yiva - yim) / yis

    # yjm, yjs = np.mean(yjtr), np.std(yjtr)
    # yjtr, yjva = (yjtr - yjm) / yjs, (yjva - yjm) / yjs

    # treat i and j as one

    ytr = []
    ytr.extend(np.squeeze(yitr).tolist())
    ytr.extend(np.squeeze(yjtr).tolist())
    #print(ytr)
    ytr = list(set(Counter(ytr)))

    ym, ys = np.mean(ytr), np.std(ytr)

    yim, yis = ym, ys
    yjm, yjs = ym, ys

    yitr, yiva = (yitr - yim) / yis, (yiva - yim) / yis
    yjtr, yjva = (yjtr - yjm) / yjs, (yjva - yjm) / yjs
    
    best_logpvalid = - np.inf

    G = tf.Graph()

    with G.as_default():
        sess = tf.InteractiveSession()

        ed.set_seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        xi_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='xi_bin')  # binary inputs
        xi_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='xi_cont')  # continuous inputs
        ti_ph = tf.placeholder(tf.float32, [M, 1])
        yi_ph = tf.placeholder(tf.float32, [M, 1])

        xj_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='xj_bin')  # binary inputs
        xj_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='xj_cont')  # continuous inputs
        tj_ph = tf.placeholder(tf.float32, [M, 1])
        yj_ph = tf.placeholder(tf.float32, [M, 1])

        xi_ph = tf.concat([xi_ph_bin, xi_ph_cont], 1)
        xj_ph = tf.concat([xj_ph_bin, xj_ph_cont], 1)

        activation = tf.nn.elu

        # CEVAE model (decoder)
        # p(z)
        zi = Normal(loc=tf.zeros([tf.shape(xi_ph)[0], d]), scale=tf.ones([tf.shape(xi_ph)[0], d]))
        zj = Normal(loc=tf.zeros([tf.shape(xj_ph)[0], d]), scale=tf.ones([tf.shape(xj_ph)[0], d]))

        # p(x|z)
        hxi = fc_net(zi, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
        logitsxi = fc_net(hxi, [h], [[len(binfeats), None]], 'px_z_bin'.format(it + 1), lamba=lamba, activation=activation)
        xi1 = Bernoulli(logits=logitsxi, dtype=tf.float32, name='bernoulli_pxi_zi')

        mui, sigmai = fc_net(hxi, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                           activation=activation)
        xi2 = Normal(loc=mui, scale=sigmai, name='gaussian_pxi_zi')


        hxj = fc_net(zj, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation, reuse=True)
        logitsxj = fc_net(hxj, [h], [[len(binfeats), None]], 'px_z_bin'.format(it + 1), lamba=lamba, activation=activation, reuse=True)
        xj1 = Bernoulli(logits=logitsxj, dtype=tf.float32, name='bernoulli_pxj_zj')

        muj, sigmaj = fc_net(hxj, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                           activation=activation, reuse=True)
        xj2 = Normal(loc=muj, scale=sigmaj, name='gaussian_pxj_zj')

        # p(t|z)
        logitszi = fc_net(zi, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        ti = Bernoulli(logits=logitszi, dtype=tf.float32)

        logitszj = fc_net(zj, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation, reuse=True)
        tj = Bernoulli(logits=logitszj, dtype=tf.float32)

        # p(yi|ti,zi,tj,zj)

        mu2_ti0 = fc_net(zi, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_ti1 = fc_net(zi, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        #yi = Normal(loc=t * mu2_ti1 + (1. - ti) * mu2_ti0, scale=tf.ones_like(mu2_ti0))

        mu2_tj0 = fc_net(zj, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation, reuse=True)
        mu2_tj1 = fc_net(zj, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation, reuse=True)
        #yj = Normal(loc=t * mu2_tj1 + (1. - tj) * mu2_tj0, scale=tf.ones_like(mu2_tj0))

        yi = Normal(
            loc=ti * mu2_ti1 + (1. - ti) * mu2_ti0 + 
         alpha * (tj * mu2_tj1 + (1. - tj) * mu2_tj0),
          scale=tf.ones_like(mu2_ti0))

        yj = Normal(
            loc = tj * mu2_tj1 + (1. - tj) * mu2_tj0 + 
            alpha * (ti *  mu2_ti1 + (1. - ti) * mu2_ti0),
          scale = tf.ones_like(mu2_tj0))

        # NETCVAE variational approximation (encoder)
        # q(t|x)
        logits_ti = fc_net(xi_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
        qti = Bernoulli(logits=logits_ti, dtype=tf.float32)

        logits_tj = fc_net(xj_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation, reuse=True)
        qtj = Bernoulli(logits=logits_tj, dtype=tf.float32)

        # q(yi|xi,ti,xj,tj)
        hqyi = fc_net(xi_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
        mu_qyi_t0 = fc_net(hqyi, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
        mu_qyi_t1 = fc_net(hqyi, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
        #qyi = Normal(loc=qti * mu_qyi_t1 + (1. - qti) * mu_qyi_t0, scale=tf.ones_like(mu_qyi_t0))


        # q(yj|xj,tj,xi,ti)
        hqyj = fc_net(xj_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation, reuse=True)
        mu_qyj_t0 = fc_net(hqyj, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation, reuse=True)
        mu_qyj_t1 = fc_net(hqyj, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation, reuse=True)
        #qyj = Normal(loc=qtj * mu_qyj_t1 + (1. - qtj) * mu_qyj_t0, scale=tf.ones_like(mu_qyj_t0))

        qyi = Normal(
            loc=qti * mu_qyi_t1 + (1. - qti) * mu_qyi_t0 + 
            alpha * (qtj * mu_qyj_t1 + (1. - qtj) * mu_qyj_t0), 
            scale=tf.ones_like(mu_qyi_t0), name="qyi")

        qyj = Normal(
            loc= qtj * mu_qyj_t1 + (1. - qtj) * mu_qyj_t0 + 
            alpha * (qti * mu_qyi_t1 + (1. - qti) * mu_qyi_t0), 
            scale=tf.ones_like(mu_qyi_t0), name="qyj")

        # q(z|x,t,y)
        inpt2i = tf.concat([xi_ph, qyi], 1)
        hqzi = fc_net(inpt2i, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muqi_t0, sigmaqi_t0 = fc_net(hqzi, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                   activation=activation)
        muqi_t1, sigmaqi_t1 = fc_net(hqzi, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                   activation=activation)
        qzi = Normal(loc=qti * muqi_t1 + (1. - qti) * muqi_t0, scale=qti * sigmaqi_t1 + (1. - qti) * sigmaqi_t0, name="qzi")


        inpt2j = tf.concat([xj_ph, qyj], 1)
        hqzj = fc_net(inpt2j, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation, reuse=True)
        muqj_t0, sigmaqj_t0 = fc_net(hqzj, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                   activation=activation, reuse=True)
        muqj_t1, sigmaqj_t1 = fc_net(hqzj, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                   activation=activation, reuse=True)
        qzj = Normal(loc=qtj * muqj_t1 + (1. - qtj) * muqj_t0, scale=qtj * sigmaqj_t1 + (1. - qtj) * sigmaqj_t0, name="qzj")

        # Create data dictionary for edward
        data = {xi1: xi_ph_bin, xi2: xi_ph_cont, yi: yi_ph, qti: ti_ph, ti: ti_ph, qyi: yi_ph,
        xj1: xj_ph_bin, xj2: xj_ph_cont, yj: yj_ph, qtj: tj_ph, tj: tj_ph, qyj: yj_ph}

        # sample posterior predictive for p(y|z,t)
        yi_post = ed.copy(yi, {zi: qzi, ti: ti_ph, zj: qzj, tj: tj_ph}, scope='yi_post')
        yj_post = ed.copy(yj, {zi: qzi, ti: ti_ph, zj: qzj, tj: tj_ph}, scope='yj_post')
        
        # crude approximation of the above, why not mean on ti or tj?
        # yi_post_mean = ed.copy(yi, {zi: qzi.mean(), ti: ti_ph, zj:qzj.mean(), tj: tj_ph}, scope='yi_post_mean')
        # yj_post_mean = ed.copy(yj, {zi: qzi.mean(), ti: ti_ph, zj:qzj.mean(), tj: tj_ph}, scope='yj_post_mean')

        # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
        # for early stopping according to a validation set

        yi_post_eval = ed.copy(yi, {zi: qzi.mean(), qti: ti_ph, qyi: yi_ph, ti: ti_ph}, scope='yi_post_eval')
        yj_post_eval = ed.copy(yj, {zj: qzj.mean(), qtj: tj_ph, qyj: yj_ph, tj: tj_ph}, scope='yj_post_eval')

        xi1_post_eval = ed.copy(xi1, {zi: qzi.mean(), qti: ti_ph, qyi: yi_ph}, scope='xi1_post_eval')
        xi2_post_eval = ed.copy(xi2, {zi: qzi.mean(), qti: ti_ph, qyi: yi_ph}, scope='xi2_post_eval')

        xj1_post_eval = ed.copy(xj1, {zj: qzj.mean(), qtj: tj_ph, qyj: yj_ph}, scope='xj1_post_eval')
        xj2_post_eval = ed.copy(xj2, {zj: qzj.mean(), qtj: tj_ph, qyj: yj_ph}, scope='xj2_post_eval')
        
        ti_post_eval = ed.copy(ti, {zi: qzi.mean(), qti: ti_ph, qyi: yi_ph}, scope='ti_post_eval')
        tj_post_eval = ed.copy(tj, {zj: qzj.mean(), qtj: tj_ph, qyj: yj_ph}, scope='tj_post_eval')

        logp_valid = tf.reduce_mean(tf.reduce_sum(yi_post_eval.log_prob(yi_ph) + ti_post_eval.log_prob(ti_ph), axis=1) +
                                    tf.reduce_sum(xi1_post_eval.log_prob(xi_ph_bin), axis=1) +
                                    tf.reduce_sum(xi2_post_eval.log_prob(xi_ph_cont), axis=1) +
                                    tf.reduce_sum(zi.log_prob(qzi.mean()) - qzi.log_prob(qzi.mean()), axis=1)
                                    + tf.reduce_sum(yj_post_eval.log_prob(yj_ph) + tj_post_eval.log_prob(tj_ph), axis=1) +
                                    tf.reduce_sum(xj1_post_eval.log_prob(xj_ph_bin), axis=1) +
                                    tf.reduce_sum(xj2_post_eval.log_prob(xj_ph_cont), axis=1) +
                                    tf.reduce_sum(zj.log_prob(qzj.mean()) - qzj.log_prob(qzj.mean()), axis=1))

        #TODO: negative sampling...
        
        # inference = ed.KLqp({zi: qzi, zj: qzj, zi: qzj, zj: qzi}, data)
        inference = ed.KLqp({zi: qzi, zj: qzj, zi: qzj}, data)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        inference.initialize(optimizer=optimizer)

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter("log", sess.graph)
        writer.close()

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xitr.shape[0] / 100), np.arange(xitr.shape[0])

        # dictionaries needed for evaluation
        # just all zero and all one vectors, we can share them for both node i and node j
        tr0, tr1 = np.zeros((xialltr.shape[0], 1)), np.ones((xialltr.shape[0], 1))
        tr0t, tr1t = np.zeros((xite.shape[0], 1)), np.ones((xite.shape[0], 1))

        # so in the prediction, we will predict ITE for node i by setting ti=1 and ti=0, respectively. We also need tj (ti) for predicting yi (yj).
        # we keep tj (ti) as it was and set ti (tj) to 1 and 0 to have this task done.

        # predict for node i:

        fi1 = {xi_ph_bin: xialltr[:, 0:len(binfeats)], xi_ph_cont: xialltr[:, len(binfeats):], ti_ph: tr1,
         xj_ph_bin: xjalltr[:, 0:len(binfeats)], xj_ph_cont: xjalltr[:, len(binfeats):], tj_ph: tjalltr}

        fi0 = {xi_ph_bin: xialltr[:, 0:len(binfeats)], xi_ph_cont: xialltr[:, len(binfeats):], ti_ph: tr0,
         xj_ph_bin: xjalltr[:, 0:len(binfeats)], xj_ph_cont: xjalltr[:, len(binfeats):], tj_ph: tjalltr}

        # predict for node j:

        fj1 = {xi_ph_bin: xialltr[:, 0:len(binfeats)], xi_ph_cont: xialltr[:, len(binfeats):], ti_ph: tialltr,
         xj_ph_bin: xjalltr[:, 0:len(binfeats)], xj_ph_cont: xjalltr[:, len(binfeats):], tj_ph: tr1}

        fj0 = {xi_ph_bin: xialltr[:, 0:len(binfeats)], xi_ph_cont: xialltr[:, len(binfeats):], ti_ph: tialltr,
         xj_ph_bin: xjalltr[:, 0:len(binfeats)], xj_ph_cont: xjalltr[:, len(binfeats):], tj_ph: tr0}

        # why do we need to do the prediction again just for the testing pairs? 
        # I did not quite get it but I follow the orginal code.

        fi1t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr1t,
         xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tjte}

        fi0t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr0t,
         xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tjte}

        fj1t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tite,
         xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr1t}

        fj0t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tite,
         xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr0t}

        # f11t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr1t,
        #  xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr1t}
        # f10t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr1t,
        #  xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr0t}
        # f01t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr0t,
        #  xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr1t}
        # f00t = {xi_ph_bin: xite[:, 0:len(binfeats)], xi_ph_cont: xite[:, len(binfeats):], ti_ph: tr0t,
        #  xj_ph_bin: xjte[:, 0:len(binfeats)], xj_ph_cont: xjte[:, len(binfeats):], tj_ph: tr0t}

        #f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
        #f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}

        #f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
        #f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}


        for epoch in range(n_epoch):
            avg_loss = 0.0

            t0 = time.time()
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
            pbar.start()
            np.random.shuffle(idx)
            for j in range(n_iter_per_epoch):
                pbar.update(j)
                batch = np.random.choice(idx, 100)

                # we obs x,t,y for training pairs
                #x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                xi_train,yi_train,ti_train = xitr[batch], yitr[batch], titr[batch]
                xj_train,yj_train,tj_train = xjtr[batch], yjtr[batch], tjtr[batch]
                # info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                #                                         x_ph_cont: x_train[:, len(binfeats):],
                #                                         t_ph: t_train, y_ph: y_train})

                info_dict = inference.update(feed_dict={xi_ph_bin: xi_train[:, 0:len(binfeats)],
                                                        xi_ph_cont: xi_train[:, len(binfeats):],
                                                        ti_ph: ti_train, yi_ph: yi_train,
                                                        xj_ph_bin: xj_train[:, 0:len(binfeats)],
                                                        xj_ph_cont: xj_train[:, len(binfeats):],
                                                        tj_ph: tj_train, yj_ph: yj_train})

                avg_loss += info_dict['loss']

            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / 100

            if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                # logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)], x_ph_cont: xva[:, len(binfeats):],
                #                                             t_ph: tva, y_ph: yva})
                logpvalid = sess.run(logp_valid,feed_dict = {xi_ph_bin: xiva[:, 0:len(binfeats)], xi_ph_cont: xiva[:, len(binfeats):],
                                                             ti_ph: tiva, yi_ph: yiva,
                                                             xj_ph_bin: xjva[:, 0:len(binfeats)], xj_ph_cont: xjva[:, len(binfeats):],
                                                             tj_ph: tjva, yj_ph: yjva})
                if logpvalid >= best_logpvalid:
                    print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                    best_logpvalid = logpvalid
                    saver.save(sess, 'models/pair-ihdp')

            if epoch % args.print_every == 0:
                # predict for the whole training
                yi0, yi1 = get_y0_y1(sess, yi_post, fi0, fi1, shape=yialltr.shape, L=1)
                yi0, yi1 = yi0 * yis + yim, yi1 * yis + yim

                yj0, yj1 = get_y0_y1(sess, yj_post, fj0, fj1, shape=yjalltr.shape, L=1)
                yj0, yj1 = yj0 * yjs + yjm, yj1 * yjs + yjm

                # I need to figure out the average of each node, so we need to retrieve which nodes are in a certain pair.
                # We have the list of node idx, so we create a data structure s.t. we can group them by the pos and calculate average

                avg_yi0, avg_yi1 = average_y(yi0,nodei_map_alltr,pos_i_alltr), average_y(yi1,nodei_map_alltr,pos_i_alltr)
                avg_yj0, avg_yj1 = average_y(yj0,nodej_map_alltr,pos_j_alltr), average_y(yj1,nodej_map_alltr,pos_j_alltr)

                score_train_i = evaluatori_train.calc_stats(avg_yi1, avg_yi0)
                rmses_train_i = evaluatori_train.y_errors(avg_yi0, avg_yi1)

                score_train_j = evaluatorj_train.calc_stats(avg_yj1, avg_yj0)
                rmses_train_j = evaluatorj_train.y_errors(avg_yj0, avg_yj1)

                score_train = merge_scores(score_train_i,score_train_j)

                # predict for the test

                yi0t, yi1t = get_y0_y1(sess, yi_post, fi0t, fi1t, shape=yite.shape, L=1)

                yj0t, yj1t = get_y0_y1(sess, yj_post, fj0t, fj1t, shape=yjte.shape, L=1)

                # assuming that the test set shares the same mean and variance with the training set in terms of y
                # therefore, we still use yim, yis and yjm, yjs to recover the value of predicted potential outcomes
                
                yi0t, yi1t = yi0t * yis + yim, yi1t * yis + yim

                yj0t, yj1t = yj0t * yjs + yjm, yj1t * yjs + yjm

                avg_yi0t, avg_yi1t = average_y(yi0t,nodei_map_te,pos_i_te), average_y(yi1t,nodei_map_te,pos_i_te)
                avg_yj0t, avg_yj1t = average_y(yj0t,nodej_map_te,pos_j_te), average_y(yj1t,nodej_map_te,pos_j_te)

                score_test_i = evaluatori_test.calc_stats(avg_yi1t, avg_yi0t)
                score_test_j = evaluatorj_test.calc_stats(avg_yj1t, avg_yj0t)

                score_test = merge_scores(score_test_i,score_test_j)

                print ("Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                      #"rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, 
                      "ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                      "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                                           #rmses_train_i[0], rmses_train_i[1], 
                                           score_test[0], score_test[1], score_test[2],
                                           time.time() - t0))

        saver.restore(sess, 'models/pair-ihdp')
        
        # y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
        # y0, y1 = y0 * ys + ym, y1 * ys + ym
        # score = evaluator_train.calc_stats(y1, y0)
        # scores[i, :] = score

        # y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
        # y0t, y1t = y0t * ys + ym, y1t * ys + ym
        # score_test = evaluator_test.calc_stats(y1t, y0t)
        # scores_test[i, :] = score_test

        yi0, yi1 = get_y0_y1(sess, yi_post, fi0, fi1, shape=yialltr.shape, L=100)
        yi0, yi1 = yi0 * yis + yim, yi1 * yis + yim

        yj0, yj1 = get_y0_y1(sess, yj_post, fj0, fj1, shape=yjalltr.shape, L=100)
        yj0, yj1 = yj0 * yjs + yjm, yj1 * yjs + yjm

        avg_yi0, avg_yi1 = average_y(yi0,nodei_map_alltr,pos_i_alltr), average_y(yi1,nodei_map_alltr,pos_i_alltr)
        avg_yj0, avg_yj1 = average_y(yj0,nodej_map_alltr,pos_j_alltr), average_y(yj1,nodej_map_alltr,pos_j_alltr)

        score_i = evaluatori_train.calc_stats(avg_yi1, avg_yi0)
        #rmses_train_i = evaluatori_train.y_errors(avg_yi0, avg_yi1)

        score_j = evaluatorj_train.calc_stats(avg_yj1, avg_yj0)
        #rmses_train_j = evaluatorj_train.y_errors(avg_yj0, avg_yj1)

        # predict for the test

        score = merge_scores(score_i,score_j)

        scores[it, :] = score

        yi0t, yi1t = get_y0_y1(sess, yi_post, fi0t, fi1t, shape=yite.shape, L=100)
        yj0t, yj1t = get_y0_y1(sess, yj_post, fj0t, fj1t, shape=yjte.shape, L=100)

        yi0t, yi1t = yi0t * yis + yim, yi1t * yis + yim
        yj0t, yj1t = yj0t * yjs + yjm, yj1t * yjs + yjm

        avg_yi0t, avg_yi1t = average_y(yi0t,nodei_map_te,pos_i_te), average_y(yi1t,nodei_map_te,pos_i_te)
        avg_yj0t, avg_yj1t = average_y(yj0t,nodej_map_te,pos_j_te), average_y(yj1t,nodej_map_te,pos_j_te)

        score_test_i = evaluatori_test.calc_stats(avg_yi1t, avg_yi0t)
        score_test_j = evaluatorj_test.calc_stats(avg_yj1t, avg_yj0t)

        score_test = merge_scores(score_test_i,score_test_j)

        scores_test[it,:] = score_test

        print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
              ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(it + 1, args.reps,
                                                                            score[0], score[1], score[2],
                                                                            score_test[0], score_test[1], score_test[2]))
        sess.close()

print('netcvae model total scores')
means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
