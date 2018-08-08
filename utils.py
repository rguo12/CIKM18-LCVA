import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
from tensorflow.contrib.layers.python.layers import initializers
from collections import defaultdict
#import types


def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False)):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print
    return y0, y1

def nodeidx2map(nodeidx):
    nodemap = defaultdict(list)
    unique_pos = {}
    for pos, idx in enumerate(nodeidx):
        if idx not in unique_pos:
            # make sure the min pos is stored in unique_pos for each node
            unique_pos[idx] = pos
        nodemap[idx].append(pos)
    return nodemap, unique_pos

def map_concatenate(map1,map2):

    new_map = {}
    for k,v in map1.iteritems():
        new_map[k] = v
    for k,v in map2.iteritems():
        if k not in new_map:
            new_map[k] = v
        else:
            if isinstance(v,list):
                new_map[k].extend(v)

    return new_map

def average_y(y,nodemap,pos):
    # calculate the average of y for each node
    # as each node appears in multiple pairs
    # then make sure the order is the same as the original one

    avg_y = np.zeros([len(y)],dtype=np.float32)
    for k,v in nodemap.items():
        avg_y[np.asarray(v)] = np.mean(y[np.asarray(v)])

    avg_y = avg_y[pos]

    return avg_y

def merge_scores(score_i,score_j):
    len_i, len_j = score_i[3], score_j[3]
    # ite, ate, pehe
    ite = np.sqrt((score_i[0]**2*len_i + score_j[0]**2*len_j)/(len_i+len_j))
    ate = (score_i[1]*len_i + score_j[1]*len_j)/(len_i+len_j)
    pehe = np.sqrt((score_i[2]**2*len_i + score_j[2]**2*len_j)/(len_i+len_j))

    return ite, ate, pehe

